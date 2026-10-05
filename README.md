# COMP90086 Workshop 10: Diffusion Models

An interactive primer and notebook walkthrough for the Workshop 10 tutorial, based on the Week 9 diffusion lecture (Ehinger & Drummond) and `worksheet10-diffusion`.

Live site: [hesamasad.github.io/week-10-diffusion-models](https://hesamasad.github.io/week-10-diffusion-models/). Open [`index.html`](index.html) locally in a browser if you prefer. The only external requests are Google Fonts, KaTeX and highlight.js. A Light / Dark / Auto switch sits in the sidebar.

## What is in it

- **A0, prologue:** from softmax classifiers \(p(y\mid x)\) through autoencoders, VAEs and GANs to many-step generation, with a one-jump versus many-steps animation.
- **A1–A7, primer:** the lecture in order, with interactive figures: a 2-D spiral diffusing, a step-by-step derivation of the closed form, the exact reverse distribution for 1-D data (why small steps are nearly Gaussian), the two ways to parameterise the U-Net, training and sampling loops, and latent, conditional and inpainting extensions.
- **§0–§5, notebook:** every code cell from the worksheet, verbatim, with worksheet/solution tabs where they differ, each beside a live version: Gaussian mixing with bell curves, the β/ᾱ schedule, both noising methods with timings, the U-Net diagram linked to its code, live activations, one training example, the cell 29 evaluation grid, and cell 32 generation.
- **Inpainting lab:** RePaint with the unconditional worksheet U-Net.
- **References:** the papers cited in the page.

## The live model

The page runs the worksheet U-Net in the browser. It is a line-for-line PyTorch port of cell 21 (2,520,154 parameters), trained with the notebook's schedule, loss, Adam optimiser, batch size and 20 epochs, then exported with an 8-bit dense layer. A plain-JavaScript forward pass matches PyTorch to about 1e-6. Sampling runs in Web Workers; one 1000-step sample takes about 13 s.

## Notes for the tutorial

- Cell 29's solution feeds `np.array([[timestep]])` (always 100) as the time input; the page has a toggle showing the fixed `timestep * i` version.
- The cell 13 text says β rises to 0.2; the code uses 0.02.
- The worksheet's cell 27 passes `MeanSquaredError` (the class) instead of `MeanSquaredError()`.

Keys: `]` / `[` next and previous section, `P` present mode, `L` light/dark.
