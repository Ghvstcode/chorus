<p align="center">
  <img src="app-icon.png" alt="Chorus icon" width="128" />
</p>

<h1 align="center"><a href="https://chorus.sh">Chorus</a></h1>

<p align="center">All the AI, on your Mac. Built by the creators of <a href="https://conductor.build">Conductor.</a></p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/54e02cd2-9532-4153-96fd-77b057409d45" alt="Chorus screenshot" />
</p>

# Getting Started

You will need:

1. NodeJS installed and on your path
2. Rust and Cargo installed and on your path (verify with `rustc --version`, `cargo --version`)
3. `imagemagick` (optional)
4. `git-lfs` (`brew install git-lfs`)
5. `pnpm` (`brew install pnpm`)

Once you have those set up, please run:

```bash
git lfs install --force
git lfs pull
pnpm run setup  # This is also our Conductor setup script
pnpm run dev    # This is also our Conductor run script
```

Vite will run on a random even-numbered port between 1422 and 1522, inclusive. HMR will run on the next port. If there's a collision, change the instance name (makes sure to rerun the setup script).
