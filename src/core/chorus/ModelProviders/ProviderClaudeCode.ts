import { IProvider } from "./IProvider";
import {
    LLMMessage,
    StreamResponseParams,
    llmMessageToString,
    readTextAttachment,
    readWebpageAttachment,
} from "../Models";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

// Types for Claude Code stream-json output
interface ClaudeCodeSystemInit {
    type: "system";
    subtype: "init";
    session_id: string;
    model: string;
    tools: string[];
}

interface ClaudeCodeAssistantMessage {
    type: "assistant";
    message: {
        content: Array<{ type: "text"; text: string } | { type: string }>;
    };
    session_id: string;
}

interface ClaudeCodeResult {
    type: "result";
    subtype: "success" | "error";
    result?: string;
    error?: string;
    session_id: string;
}

type ClaudeCodeStreamMessage =
    | ClaudeCodeSystemInit
    | ClaudeCodeAssistantMessage
    | ClaudeCodeResult;

interface TauriStreamEvent {
    type: "data" | "error" | "stderr" | "done";
    data?: string;
    error?: string;
    exitCode?: number;
}

/**
 * Check if Claude Code CLI is available and authenticated
 */
export async function checkClaudeCodeAvailable(): Promise<{
    available: boolean;
    version: string | null;
    authenticated: boolean;
}> {
    try {
        const result = await invoke<{
            available: boolean;
            version: string | null;
            authenticated: boolean;
        }>("check_claude_code_available");
        return result;
    } catch {
        return { available: false, version: null, authenticated: false };
    }
}

export class ProviderClaudeCode implements IProvider {
    async streamResponse({
        llmConversation,
        modelConfig,
        onChunk,
        onComplete,
        onError,
    }: StreamResponseParams): Promise<void> {
        // Generate a unique request ID for this stream
        const requestId = crypto.randomUUID();

        // Convert conversation to a prompt string
        // For multi-turn conversations, we format them as a single prompt with context
        const prompt = await this.formatConversationAsPrompt(llmConversation);

        // Determine model to use (extract from modelId like "claude-code::opus")
        const modelPart = modelConfig.modelId.split("::")[1];
        const model = this.mapModelName(modelPart);

        // Set up event listener for streaming responses
        let unlisten: UnlistenFn | undefined;
        let hasCompleted = false;

        try {
            // Listen for stream events from Tauri
            unlisten = await listen<TauriStreamEvent>(
                `claude-code-stream-${requestId}`,
                (event) => {
                    const payload = event.payload;

                    if (payload.type === "data" && payload.data) {
                        try {
                            const message = JSON.parse(
                                payload.data,
                            ) as ClaudeCodeStreamMessage;
                            this.handleStreamMessage(message, onChunk);
                        } catch {
                            // Not valid JSON, might be partial output
                            console.warn(
                                "Failed to parse Claude Code stream data:",
                                payload.data,
                            );
                        }
                    } else if (payload.type === "error") {
                        if (!hasCompleted) {
                            hasCompleted = true;
                            onError(payload.error || "Unknown error");
                        }
                    } else if (payload.type === "done") {
                        if (!hasCompleted) {
                            hasCompleted = true;
                            void onComplete();
                        }
                    }
                },
            );

            // Start the Claude Code CLI process
            // By default, disable project context so Claude acts as a general assistant
            await invoke("stream_claude_code_response", {
                requestId,
                prompt,
                systemPrompt: modelConfig.systemPrompt || undefined,
                model,
                disableProjectContext: true, // Run as general assistant, not project-aware
            });

            // Wait for completion (the done event will trigger onComplete)
            // We need to wait here to keep the listener active
            await new Promise<void>((resolve) => {
                const checkInterval = setInterval(() => {
                    if (hasCompleted) {
                        clearInterval(checkInterval);
                        resolve();
                    }
                }, 100);

                // Timeout after 5 minutes
                setTimeout(() => {
                    if (!hasCompleted) {
                        hasCompleted = true;
                        clearInterval(checkInterval);
                        onError("Request timed out after 5 minutes");
                        resolve();
                    }
                }, 5 * 60 * 1000);
            });
        } finally {
            // Clean up the event listener
            if (unlisten) {
                unlisten();
            }
        }
    }

    private handleStreamMessage(
        message: ClaudeCodeStreamMessage,
        onChunk: (chunk: string) => void,
    ): void {
        if (message.type === "assistant" && message.message?.content) {
            // Extract text from content blocks
            for (const block of message.message.content) {
                if (block.type === "text" && "text" in block) {
                    onChunk(block.text);
                }
            }
        }
        // We ignore "system" init messages and "result" messages
        // The "result" contains the final text but we've already streamed it
    }

    private mapModelName(modelPart: string | undefined): string | undefined {
        // Map our model identifiers to Claude CLI model names
        switch (modelPart) {
            case "opus":
            case "opus-4.5":
                return "opus";
            case "sonnet":
            case "sonnet-4.5":
                return "sonnet";
            case "haiku":
                return "haiku";
            default:
                // Let Claude CLI use its default
                return undefined;
        }
    }

    private async formatConversationAsPrompt(
        messages: LLMMessage[],
    ): Promise<string> {
        // For single message, just return the content
        if (messages.length === 1 && messages[0].role === "user") {
            return await this.formatSingleMessage(messages[0]);
        }

        // For multi-turn conversations, format as a transcript
        const parts: string[] = [];

        for (const message of messages) {
            const formatted = await this.formatMessageForTranscript(message);
            parts.push(formatted);
        }

        return parts.join("\n\n");
    }

    private async formatSingleMessage(message: LLMMessage): Promise<string> {
        if (message.role !== "user") {
            return llmMessageToString(message);
        }

        let content = message.content;

        // Append attachment contents
        // Note: The Claude Code CLI in print mode doesn't support direct image/PDF input.
        // Images would require either keeping tools enabled (to use Read tool on files)
        // or using the stream-json input format with base64 data.
        // For now, we only support text and webpage attachments which can be inlined.
        for (const attachment of message.attachments) {
            if (attachment.type === "text") {
                const textContent = await readTextAttachment(attachment);
                content += `\n\n[Attachment: ${attachment.originalName}]\n${textContent}`;
            } else if (attachment.type === "webpage") {
                const webContent = await readWebpageAttachment(attachment);
                content += `\n\n[Webpage: ${attachment.originalName}]\n${webContent}`;
            } else if (attachment.type === "image" || attachment.type === "pdf") {
                // Images and PDFs are not supported in CLI print mode without tools
                content += `\n\n[Attachment: ${attachment.originalName}] (Note: ${attachment.type} attachments are not supported with Claude Code in general assistant mode)`;
            }
        }

        return content;
    }

    private async formatMessageForTranscript(
        message: LLMMessage,
    ): Promise<string> {
        if (message.role === "user") {
            const content = await this.formatSingleMessage(message);
            return `Human: ${content}`;
        } else if (message.role === "assistant") {
            return `Assistant: ${message.content}`;
        } else if (message.role === "tool_results") {
            return `Tool Results: ${llmMessageToString(message)}`;
        }
        return "";
    }
}
