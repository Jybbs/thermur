# The Visual Identity

*Thermur* draws its site, its charts, and its marks in one identity, whose theme is fire, weather, and the flock. The palette leads with the fire of the logo over the page's own ink, the heat ramp colors temperature and a link's fire, and the marks lay the logo's birds along a gradient that runs from the ink to the fire. The package declares the identity once in `thermur.identity`, and every table below is written from that declaration, so a value copied from here onto a slide or into a figure drawn outside the repository is the value the site and the charts read.

## The Palette

A role is one use of color, such as the page's ground, its text, its links, or a chart's second series, and each role takes one color under the light scheme and one under the dark. Body text and every role set in text hold the 4.5:1 contrast [WCAG 2.2](https://www.w3.org/TR/WCAG22/) sets for text against each ground they sit on, and a mark that carries meaning holds the 3:1 it sets for a graphical object. A chart assigns its series in the order below and never cycles them, so a series past the last folds into an *Other* group or a second chart.

<!-- palette -->

| **Role** | **Light** | **Dark** | **Holds** |
|---|---|---|---|
| Critical | Pompeian Red `#a82a38` | Pompeian Red `#a82a38` | *The color of a failed state, always beside an icon and a label* |
| Ground | Cararra `#f2f1ee` | Woodsmoke `#17171b` | *The color of the page and every chart surface* |
| Ink | Woodsmoke `#17171b` | Cararra `#f2f1ee` | *The color of text, axes, and every label on a chart* |
| Lead | Fire Lord Deep `#995500` | Fire Lord Lift `#c38243` | *The site's accent and the color of its links* |
| Warning | Eureka `#e8c840` | Eureka `#e8c840` | *The color of a state worth attention, always beside an icon and a label* |
| Chart 1 | Fire Lord `#bb7733` | Fire Lord Lift `#c38243` | *Series 1 of a chart* |
| Chart 2 | Toronto `#5069ad` | Toronto Lift `#657ec1` | *Series 2 of a chart* |
| Chart 3 | Topiary Garden `#6f7c00` | Topiary Garden Lift `#839230` | *Series 3 of a chart* |
| Chart 4 | Purpureus `#9a4eae` | Purpureus Lift `#ae66c2` | *Series 4 of a chart* |
| Chart 5 | Viridian `#1e9167` | Viridian Lift `#43a67d` | *Series 5 of a chart* |

<!-- /palette -->

The table is written from `Palette` in `thermur.identity` by `mise render`.

## The Heat Ramp

The heat ramp colors temperature, from its first step, the lightest and the hottest, to its last, the darkest, and lends its steps to the fire a link burns with under the pointer, on keyboard focus, and on a tap. Critical takes one step of the ramp as its color, so a failed state and that step of the heat share one color.

<!-- heat -->

| **Step** | **Color** |
|---|---|
| 1 | Kindleflame `#e9967a` |
| 2 | Emberglow `#ea6759` |
| 3 | Autumn Fire `#c44e4f` |
| 4 | Pompeian Red `#a82a38` |
| 5 | Flame Red `#86282e` |
| 6 | Burnt Crimson `#582124` |

<!-- /heat -->

The ramp is written from the `heat` field of `Palette` by `mise render`.

## The Faces

The site sets its headings and the wordmark in a display face, its running text in a body face, and its code, readings, and every control's label in a mono face, each loaded from [Google Fonts](https://fonts.google.com/) at the weights below. Where a face does not load, a reader's browser falls back through the stack beside it.

<!-- faces -->

| **Face** | **Sets** | **Weights** | **Styles** | **Stack** |
|---|---|---|---|---|
| Crimson Pro | *The face of running text and every caption* | 400, 500, 600, 700 | Roman and italic | `'Crimson Pro', Georgia, serif` |
| Bellefair | *The face of the wordmark, page titles, and headings* | 400 | Roman | `'Bellefair', Didot, Georgia, serif` |
| JetBrains Mono | *The face of code, file paths, readings, and numbers that line up in columns* | 400, 500 | Roman | `'JetBrains Mono', ui-monospace, Menlo, monospace` |

<!-- /faces -->

The table is written from `Faces` in `thermur.identity` by `mise render`.

## The Tagline

The tagline sits beneath the wordmark on the front page, set in the body face's italic in a lighter share of the page's ink.

<!-- tagline -->

> *"A starling murmuration that breaks around a wildfire the way it breaks around a falcon."*

<!-- /tagline -->

The line is written from `Tagline` in `thermur.identity` by `mise render`.

## The Marks

The mark lays every bird of the original logo along the flock's gradient, and the small mark keeps only its larger birds for a favicon or a badge. The wordmark sets the name in the display face, its last letters breaking into birds that warm toward the fire. Each is drawn once for a light page and once for a dark one, and each file sits under `docs/public/`.

<!-- marks -->

| **Mark** | **Light** | **Dark** | **Holds** |
|---|---|---|---|
| Mark | `mark-light.svg` | `mark-dark.svg` | *The mark, every bird of the original logo along the flock's gradient* |
| Small | `mark-small-light.svg` | `mark-small-dark.svg` | *The mark drawn below 40 pixels, keeping only the original's larger birds, for a favicon or a badge* |
| Wordmark | `wordmark-light.png` | `wordmark-dark.png` | *The wordmark, the name set in the display face breaking into birds* |

<!-- /marks -->

The table is written from `Marks` in `thermur.identity` by `mise render`.
