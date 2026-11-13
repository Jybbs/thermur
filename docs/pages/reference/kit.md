# The Kit

Every widget on the site builds its controls, the chrome around its drawing, and the markup a chart writes from one kit, the `@thermur/kit` package under `docs/kit/`, so a slider, a group of radios, or a menu works the same way on every page. Each control runs on a [Zag.js](https://zagjs.com/) machine, which carries the keys, the focus, and the names a screen reader hears from the [ARIA Authoring Practices](https://www.w3.org/WAI/ARIA/apg/patterns/) for that pattern, while the kit sets only its look in the site's identity. A drawing the site paints into a canvas or an SVG keeps drawing its own track or stop, and the kit's control stands over it, so a reader's drag, press, or key reaches the same machine either way.

## The Pieces

The catalog below walks the kit one piece at a time, its stage setting out each part of the kit at work on a made-up fire front and flock, and the card beneath it naming when a widget reads the piece, how many of the site's modules read it, what it replaces, and the line that takes it up. A control sets its label in the mono face at weight 500, uppercase, with 0.1em of letter spacing, and every transition reads the site's shared settling time, which drops to zero for a reader who prefers reduced motion.

<!-- pieces -->

| **Piece** | **Imports** | **Machine** | **A Widget Reads It When** | **It Replaces** |
|---|---|---|---|---|
| Chip | `Chip` | None | A drawing reads a value out beside its name, as the threat's front reads the time and the neighbors read the seventh's distance | The threat's outlined chip and the filled chip the neighbors' header built by hand |
| Copy | `heard`, `useCopy` | None | A press copies a value to the clipboard, as each color, face, and step of the heat ramp on The Visual Identity does | The copier the identity's widgets read from the specimen helpers, the words each of them set out of sight for a screen reader, and the fuse The Visual Identity's sheet burned beneath a copied value |
| Radios | `Radios` | `radio-group` | A reader picks one of a few choices, as the neighbors pick the pointer's mode and the heading chart picks the first drone's turn on its rim | The radio group the neighbors' header built by hand and the radios the heading chart drew into its dial |
| Select | `Select` | `select` | A reader picks one of a list of choices inside a sentence, as the Data page sets how much of an output the graph reads | The native menus of the crop's sentence, and the measuring that held the sentence at its widest wording |
| Slider | `Slider` | `slider` | A reader sets one number along a range, on its own track or over a track a drawing draws, stepping between stops where a drawing has them | The native range inputs of the threat's rail, the neighbors' range, the marks' dusk, and the faces' time, beside the handle the coupling chart drew into its track |
| Toggle | `Toggle` | `toggle` | A reader turns a state on and off with one button, as the page's pause and the coupling chart's play knob do | The button the page's pause set its own pressed state on and the knob the coupling chart drew to play its sweep |
| Markup | `markup` | None | A chart writes its SVG, setting its colors and faces from the identity's custom properties, and draws again at each width it takes | The SVG helpers of the charts' own module and the width watch of their motion helpers |
| The Gate | `onMotion`, `pause`, `paused`, `still` | None | A drawing moves on its own and stands still while the page's pause is on or for a reader who prefers reduced motion | The pause helpers under the canvas folder, the reduced-motion query each stage made of its own, and the motion helpers the charts read |
| The Loop | `Frames`, `useLoop` | None | A drawing moves on its own frame by frame, starting only while the gate lets motion run and stopping as the gate stands motion still | The frame loop the coupling chart wrote by hand for its sweep, and the one the stage behind every canvas drawing ran |
| Timing | `idle`, `tween` | None | A drawing eases from one state to the next over the theme's settling time, holding idle as what stops it while none runs | The transition the charts' motion helpers ran, and the do-nothing stopper each chart declared for itself |

<!-- /pieces -->

The table is written by `mise render` from the declaration each piece keeps beside its code under `docs/kit/src/`.
