# Logo analysis

The supplied `hdfc-logo.png` is a 900 × 900 indexed PNG with a 13-color palette.
It has **no alpha channel**: its gray/white checkerboard is baked into the image,
not a transparency preview. The original asset is preserved unchanged.

The mark consists of four red right-angle corners framing a central blue square.
The white, bold HDFC BANK wordmark sits on a navy band. Its main source colors are
red `#FE0000` / `#FF0000`, blue `#003297`, navy `#001D78` and white.

`public/brand-mark.svg` renders that geometric mark as a small, sharp vector so
the checkerboard is not carried into the interface. The header pairs it with a
bold wordmark. The interface extends the palette with a deeper navy navigation
surface (`#0B1F3B`), a softer red action color (`#E52B3D`), cool white panels and
restrained green for successful verification. Inter is bundled locally so text
does not depend on a third-party font service.

The progress UI uses actual server events. Waiting stages stay neutral; elapsed
time is not an ETA, and completed-stage counts are not a time-based percentage.
