# The Interaction Rules

Each drone in *Thermur* steers by the sum of a few rules, and each rule reads only what that one drone senses. The controller has a slot for each part of the motion, meaning the heading a drone turns toward, the rate it aligns with each neighbor at, the random increments driving it, the force holding it at a spacing, the contact keeping two drones apart, the roost holding the flock over one place, and the speed control holding it near the cruise speed. Each slot holds a rule borrowed from another physical system, a starling flock, a school of fish, or a swarm of active particles. An open slot starts from a default and flies each candidate rule against it, keeping a candidate only where the measurements favor it. In the flights below the elastic springs and the fitted attraction pass that test for the cohesion, the anisotropic and the distance-weighted rates for the coupling, and the delayed and the overdamped rules for the heading, and every slot still holds its default.

## The Neighbors a Drone Senses

A drone senses its seven nearest neighbors by rank rather than within a fixed range, the topological interaction measured in starling flocks[^ballerini2008a]. For each neighbor it reads the offset, the distance, and the velocity as its radio last heard them, together with the velocity a neighbor held half a second before that for the rule that reads a delay, and it reads its own position, its own command, and the excess temperature and the wind of the air at its position. A rule receives that view and nothing else.

By rank a drone always senses seven neighbors, whereas the number within a fixed range swings as a flock gathers and splits. The drawing below flies the front page's flock under a TypeScript copy of the default composite, whose step and radio the site's suite checks against the package's own from the same states and draws. The pointer acts as a falcon through a push the drawing adds beside the composite, whose default slots read no air. The marked bird's seven are the senders it last heard, ranked by where its radio heard them, whereas the ring counts the birds within the fixed range by the distance between the drones the drawing flies.

<!-- neighbors -->

The default cohesion reads one quantity past the seven, the drone's centrality, over every sender its radio holds. The suite checks this with a property test, which moves a drone that another drone cannot hear to a random place still out of that drone's radio range and asserts that the second drone's acceleration and the amplitude of its noise stay the same, under the default composite and under every candidate.

## The Flight

A flight integrates each drone's command under the sum of the slots, and its velocity and position under the plant and the wind below, through the Euler-Maruyama scheme, one step of diffrax's Euler solver at a time, its noise the Brownian increment a virtual Brownian tree draws, which carries the $`\sqrt{\Delta t}`$ scaling of a stochastic differential equation[^higham2001]. Every draw derives from one seed, and several seeds fly together as one batch. The flock launches in a thin horizontal cloud, its axes in the proportion 5.6 to 2.8 to 1 measured on starling flocks, its thin axis vertical[^ballerini2008b]. Under a heading rule that holds the speed, the part of the acceleration along a drone's command is removed, so the command keeps the cruise speed, whereas under the default heading rule the speed is free and the speed slot relaxes it toward the cruise speed.

The table is written from the record each flight reads, set under `[tool.thermur.plan]` or through a `--plan.` flag of `thermur fly`, and `mise render` rewrites it.

<!-- plan -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `axes` | `[5.6, 2.8, 1.0]` |  | The launch cloud's length, width, and depth, in proportion, the thin horizontal shape measured on starling flocks, whose long axis holds no fixed bearing to their motion, launched with its long axis along the course and its thin axis vertical |
| `center_m` | `[1000.0, 1000.0, 100.0]` |  | The center of the launch cloud, east, north, and above the ground, in meters, by default on the parametric threat's course to the front it crosses within two minutes |
| `course_deg` | `180.0` |  | The launch heading in degrees counterclockwise from east, by default west |
| `cruise_m_s` | `10.0` | above 0 | The speed a flight holds each drone at, in meters per second |
| `drones` | `100` | at least 1 | The size of the flock |
| `duration_s` | `120.0` | above 0 | The length of the flight, in seconds |
| `frame_every` | `25` | at least 1 | The number of steps between two kept frames |
| `neighbors` | `7` | at least 1 | The number of topological neighbors each drone senses |
| `resolution_s` | `0.005` | above 0 | The time resolution of the Brownian path, in seconds, which a flight at a smaller `step_s` replays unchanged |
| `scatter_deg` | `10.0` | at least 0 and below 90 | The spread of the launch headings about the course, in degrees |
| `seed` | `0` |  | The value every draw of a flight derives from |
| `seeds` | `4` | at least 1 | The number of launches flown together under `vmap` |
| `settled_s` | `20.0` | at least 0 | The start of the flight a statistic leaves out, in seconds |
| `shape_every_s` | `1.0` | above 0 | The time between the frames a flight's shape is averaged over, in seconds, by default the second between the measurements a study of the starling model averages a flock's shape over |
| `spacing_m` | `15.0` | above 0 | The launch cloud's cube root of the volume per drone, in meters |
| `starling_m` | `0.69` | above 0 | The nearest-neighbor distance of the starling flock the controller's values are taken at, in meters, by default that of the starling model's flock of 781 birds |
| `step_s` | `0.02` | above 0 | The step of the integrator, in seconds |
| `turn_at_s` | `25.0` | at least 0 | The moment the first drone starts the turn, in seconds |
| `turn_deg` | `0.0` |  | The angle the first drone turns through and then holds, in degrees counterclockwise, none by default |
| `turning_deg_s` | `2.0` | above 0 | The rotation of the flock's mean heading, over `turning_window_s`, past which a spontaneous turn counts as started, in degrees per second |
| `turning_window_s` | `5.0` | above 0 | The time over which the flock's mean heading or a drone's heading is read for a turn, in seconds, longer than a step's noise on either |

<!-- /plan -->

## The Starling Scale

Each default the slots hold is a value a starling study states at a starling's scale, and a flight carries every one of them to the drones' scale through one factor. The drone flock is taken as the starling flock magnified at the cruise speed of 10 m/s the starling model flies[^hildenbrandt2010], by the launch's nearest-neighbor distance over the 0.69 m that model's flock of 781 birds holds. The launch's nearest-neighbor distance is the mean distance points at random hold at a given density[^hertz1909], 0.554 times the cube root of the volume per point, averaged over the density of the Gaussian cloud the plan launches, which at a launch spacing of 15 m gives 23.7 m and a factor of 34.4. At one speed a time scales by the same factor as a length, so a slot multiplies each length by the factor, divides each acceleration and each rate by it, divides a spring's stiffness by its square, and divides a noise's amplitude by its square root, which keeps the spread the noise drives in proportion to the spacing. `Composite.from_settings` scales every slot, so a value set under `[tool.thermur]` or through a flag reads at a starling's scale, whereas each candidate a trial flies holds the value its table gives.

## The Plant and the Wind

A drone's autopilot carries its velocity through the air toward its command over a first-order lag of one second[^viragh2014], holding the command within the speeds and the response within the accelerations the PX4 multicopter defaults set, with the horizontal speed raised within PX4's range to the 15.2 m/s of the fastest starling flock measured[^ballerini2008b], and the descent's speed and acceleration raised within PX4's ranges to match the climb's. A bound tighter below than above turns noise of zero mean on a vertical command into a mean climb, which under PX4's own descent bounds lifts the default flight to a mean height of 160 m over two minutes against the roost's 100 m. The plant holds no gravity term, since the autopilot carries the drone's weight, and it holds the drone's own bounds, so the starling scale leaves it as it is. A drone moves over the ground at its velocity through the air plus the wind at its position[^berman2021], so the WRF-SFIRE winds carry the flock in place of a drag force, whereas the parametric threat's air is still. The table is written from the plant's record, set under `[tool.thermur.plant]` or through a `--plant.` flag, and `mise render` rewrites it.

<!-- plant -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `climb_m_s` | `3.0` | above 0 | The fastest climb a command holds, in meters per second, PX4's `MPC_Z_VEL_MAX_UP` |
| `climbing_m_s2` | `4.0` | above 0 | The largest upward acceleration, in meters per second squared, PX4's `MPC_ACC_UP_MAX` |
| `descending_m_s2` | `4.0` | above 0 | The largest downward acceleration, in meters per second squared, PX4's `MPC_ACC_DOWN_MAX`, raised from its default of 3 within the 2 to 15 PX4 allows to match `climbing_m_s2` |
| `descent_m_s` | `3.0` | above 0 | The fastest descent a command holds, in meters per second, PX4's `MPC_Z_VEL_MAX_DN`, raised from its default of 1.5 within the 0.5 to 4 PX4 allows to match `climb_m_s` |
| `lag_s` | `1.0` | above 0 | The time constant over which a drone's velocity relaxes toward its command, in seconds |
| `level_m_s` | `15.2` | above 0 | The fastest horizontal speed a command holds, in meters per second, within the 0 to 20 PX4 allows its `MPC_XY_VEL_MAX` |
| `level_m_s2` | `5.0` | above 0 | The largest horizontal acceleration, in meters per second squared, the ceiling PX4's `MPC_ACC_HOR_MAX` sets in its direct-velocity mode |

<!-- /plant -->

## The Radio

A drone hears its neighbors only through its radio, which broadcasts each drone's state five times a second and delivers it one second late, the sensor update period and the delay of the setup the outdoor drone flock was simulated under[^vasarhelyi2018], whose update period matches the earlier quadcopter flock's[^viragh2014]. A packet arrives only where its sender sat within 50 m of the drone as it was sent, the distance past which the outdoor flock's radios began to lose long runs of packets, nearly every packet arriving from a peer within 20 m. The radio is a drone's own, so the starling scale leaves its delay, its rate, its loss, and its range as they are. A drone ranks its seven nearest neighbors by the positions it heard, as a starling interacts with a fixed number of the neighbors it perceives[^ballerini2008a], so a neighbor heard from within range may since have flown past it, and a drone that has heard fewer than seven others reads each missing neighbor as one sitting on the drone at its own command. Each packet from each sender is lost with a chance of 0.4, the middle of the 20 to 60% the outdoor flock's radios lost, and a drone keeps the last state it heard from a sender until another packet arrives. A drone hears a sender for three packet periods after its last packet, 0.6 s, and then forgets it, the neighbor hold time of three refresh intervals that the routing protocols of mobile ad hoc networks set[^clausen2003][^clausen2011]. Each packet carries the sender's position, its velocity, and the state of its noise slot. The replay that scores each composite's drift delivers every packet sent from within range, keeping the delay. The table is written from the radio's record, and `mise render` rewrites it.

<!-- radio -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `delay_s` | `1.0` | at least 0 | The age of a broadcast as it arrives, in seconds |
| `hold_periods` | `3` | at least 1 | The number of packet periods after its last packet for which a receiver keeps hearing a sender, by default the three refresh intervals the mobile ad hoc routing protocols OLSR and NHDP set as the neighbor hold time |
| `loss` | `0.4` | at least 0 and below 1 | The share of the packets between two drones the link loses, each loss drawn apart from every other |
| `range_m` | `50.0` | above 0 | The distance between two drones, as a packet is sent, past which the packet never arrives, in meters, by default the distance past which the outdoor drone flock's radios began to fail |
| `rate_hz` | `5.0` | above 0 | The number of packets each drone sends each second |

<!-- /radio -->

## The Roost

The roost slot keeps the flock flat through the vertical part of the starling model's roost term[^hildenbrandt2010], a pull toward the roost's height in proportion to the height away, and against the drone's climb over the ground at twice the square root of that pull's stiffness, which damps the height critically, since the starling model tunes its speed's relaxation with that pull so its flocks fly level when nothing turns them, through a lift that rises with speed, and the drone's plant carries no such lift. That model's vertical weight over its bird's mass, 0.2 N per meter over 80 g, reads per meter because the model multiplies it by the height away, although its table of parameters lists it in newtons, and the starling scale carries its 2.5 per square second to 0.0021. The height defaults to the launch's 100 m. The slot also holds the flock over a disk of 1000 m radius around the launch's center, at (1000, 1000) m inside the parametric threat's 4 km by 2 km grid, through the turn fish in a circular tank take from its wall[^gautrais2012]. A drone heading for the rim turns across its heading, away from the rim, at a rate of 0.94 over its time to reach the rim at its ground speed, the strength fitted to the fish, so the turn grows as the rim nears. The slot applies that rate as an acceleration across the heading at the ground speed, capped at the plant's horizontal bound of 5 m/s², whereas the fish relax their turning rate toward it, and a drone outside the disk turns at the cap. The strength is a ratio of times, so the starling scale leaves it as it is, and the radius is set at the drone's scale. The table is written from the slot's record, and `mise render` rewrites it.

<!-- roost -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `bound_m_s2` | `5.0` | above 0 | The largest turning acceleration, in meters per second squared, by default the plant's bound on a horizontal acceleration |
| `center_m` | `[1000.0, 1000.0]` |  | The disk's center east and north, in meters, by default the launch's |
| `height_m` | `100.0` |  | The height the drone is pulled toward, in meters above the ground, by default the launch's |
| `radius_m` | `1000.0` | above 0 | The disk's radius, in meters |
| `stiffness_per_s2` | `2.5` | at least 0 | The acceleration toward the height per meter away from it, in reciprocal square seconds, at a starling's scale, by default the starling model's vertical weight of 0.2 newtons per meter over its bird's mass of 80 grams, read per meter since the model multiplies it by the height away, although its table of parameters lists it in newtons |
| `turn` | `0.94` | at least 0 | The turn rate times the time to reach the rim, in radians, by default the strength fitted to fish turning from the wall of their tank |

<!-- /roost -->

## The Speed

The speed slot relaxes a drone's speed along its heading toward the cruise speed through the starling model's speed control[^hildenbrandt2010], so a drone left behind in a turn catches up and one pushed ahead falls back. At a starling's scale the speed relaxes over 1 s toward 10 m/s, the model's own values, which a later study of that model flies as well[^hemelrijk2011], and the starling scale carries the second to 34.4 s. The control acts only under a heading rule that leaves the speed free, as the default does, since a rule holding the speed removes every part of the acceleration along the command. The table is written from the slot's record, and `mise render` rewrites it.

<!-- speed -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `cruise_m_s` | `10.0` | above 0 | The speed the drone relaxes toward, in meters per second, by default the model's cruise speed of 10 meters per second, which the plan launches the flock at |
| `relaxation_s` | `1.0` | above 0 | The time over which the drone's speed relaxes toward `cruise_m_s`, in seconds, at a starling's scale, by default the model's speed control of 1 second |

<!-- /speed -->

Under the defaults, 100 drones over four seeds keep their nearest neighbors 15.0 m away at 120 s and 22.6 m at 300 s on the mean, inside the 12 to 30 m the outdoor flock kept[^vasarhelyi2018], at a polarization of 0.72 to 0.84 and 0.73 to 0.86 across the seeds, below the 0.96 of starling flocks. Every drone stays over the roost's disk at every frame, and the settled flock's mean height is 99.3 m and 99.4 m, single drones ranging from 28 m to 152 m at 120 s and from 28 m to 154 m by 300 s. A drone is zigging, waiting to, or in its refractory time in 36% and 31% of the frames. A flock's shape is averaged over time, as the starling model's flocks are, each property measured once per second over 30 minutes[^hemelrijk2011]. Read each second past the first 20 s, the largest group the flock splits into stands at 1 to 2.26 to 6.41 through 120 s and 1 to 1.87 to 6.31 through 300 s on the mean over the seeds, so at both lengths its longest axis falls inside the span of the starling flocks the starling model is checked against[^hildenbrandt2010], whose longest axes run 4.02 to 6.94 times their thinnest. The shape check compares the longest axis alone, because the longest axis over the thinnest does not depend on flock size in the flocks over Rome behind that span, of 448 to 2631 birds, nor in the starling model's flocks matched to them, whereas the model's middle axis over the thinnest rises with flock size, a dependence the study lists among the model's mismatches with the flocks over Rome[^hildenbrandt2010]. Counting two drones in one group where a chain of hops under 45 m joins them, the flock flies in 7, 5, 6, and 4 groups across the seeds at 120 s and 8, 7, 4, and 2 at 300 s, its largest group holding 58 to 99 of the drones at 300 s. At 1000 drones, read each second past the first 20 s through 300 s, the fewest drones the largest group holds runs from 490 to 633 across the seeds, and on the mean its middle axis stands at 2.92 to 3.43 times its thinnest, inside the Rome flocks' 1.90 to 3.58, while its longest runs 11.5 to 13.0 times, about twice their 4.02 to 6.94.

## Contact

Two drones push apart through the separation of the starling model[^hildenbrandt2010], a Gaussian in the distance past a hard sphere, averaged over the neighbors the drone has heard and directed away from each. At a starling's scale the hard sphere's radius is 0.2 m, half a starling's wingspan, the Gaussian's standard deviation of 0.461 m leaves the push at a hundredth of its strength at the separation radius of 1.6 m the model sets for its flock of 781 birds, and the push at full strength is the model's 1 N over its bird's 80 g, which the starling scale carries to 6.9 m, 15.9 m, and 0.36 m/s². The model's equation prints the Gaussian as $`\exp(-(d - r_h)^2 / \sigma^2)`$, whereas its text calls $`\sigma`$ a standard deviation, and the rule takes the text's reading. The contact is fixed rather than open. The table is written from the rule's record, and `mise render` rewrites it.

<!-- contact -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `sphere_m` | `0.2` | above 0 | The radius of the hard sphere, in meters, at a starling's scale, by default half a starling's wingspan |
| `spread_m` | `0.461` | above 0 | The standard deviation of the Gaussian past the hard sphere, in meters, at a starling's scale, by default the one leaving the push at a hundredth of its strength at the separation radius of 1.6 meters the model sets for its flock of 781 birds, `(1.6 - 0.2) / sqrt(2 * ln 100)` |
| `weight_m_s2` | `12.5` | at least 0 | The push at full strength, in meters per second squared, at a starling's scale, by default the model's separation weight of 1 newton over its bird's mass of 80 grams |

<!-- /contact -->

## Heading

The default heading rule is a spinless second-order consensus[^lizarraga2026], which relaxes a drone's velocity toward each neighbor's at the rate the coupling sets and toward the neighbors' mean as well, where the source relaxes toward the whole flock's, and leaves the speed to the speed slot. Its pull toward the mean is the source's aggregation strength over its alignment strength per neighbor, 0.01 to 50, times the coupling's rate, so it carries to the drones' scale as the coupling does. The candidates are the overdamped limit of the inertial spin model at a constant speed[^cavagna2015], which holds the speed and carries no spin between steps, and alignment on the neighbors' velocities half a second old[^geiss2022]. The trial turns one drone through 20° and through 90° and measures how far the turn reaches across the flock and how fast it crosses, beside the polarization and the frequency at which the spontaneous velocity correlation $`C(k,\omega)`$ peaks. The table is written from each rule's record, and `mise render` rewrites it.

<!-- heading -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `consensus.centering` | `0.0002` | at least 0 | The rate at which a drone's velocity relaxes toward its neighbors' mean, as a share of the coupling's mean rate per neighbor, by default the source's aggregation strength of 0.01 over its alignment strength of 50 per neighbor, so it carries to any scale the coupling does |
| `delayed.lag_s` | `0.5` | above 0 | The age of the neighbor states the slot reads, in seconds |

<!-- /heading -->

The chart below reads the share of the first drone's turn that the farthest of five shells of drones turned through under each heading rule, the mean across four seeds beside its standard error, from a run of `thermur fly coupling_trial heading_trial` that `mise measure` wrote out, and `mise measure` redraws it from a newer run.

<!-- reach -->

Under every heading rule the farthest drones follow more than half of a 20° turn and under a sixth of a 90° one, the default reaching 0.58 and 0.13, the delayed rule 0.68 and 0.15, and the overdamped rule 0.72 and 0.16. The standard errors run from 14% to 18% of the means, so four seeds leave the three rules within error of one another on either turn.

## Coupling

The default coupling holds one constant rate $`J`$ over every one of a drone's seven neighbors[^bialek2012], at a starling's scale the rate at which a starling's heading relaxes toward its neighbors', inferred from the dynamics of starling flocks[^mora2016] and spread over the seven, $`1 / (7 \cdot 0.158)`$ or 0.904 per second, where 0.158 s is the geometric mean of the flocks' relaxation times, which the starling scale carries to 0.0263. The candidates are a rule that drops each of a drone's seven neighbors past a fixed range once the air at the drone runs hot, after the interactions jackdaws show while mobbing a predator[^ling2019b], a rate that decays with distance even within a rank, after the interactions by physical distance rather than by rank that chimney swifts show[^evangelista2017], and a rate stronger for a neighbor flying abeam than for one ahead or behind, carrying over to alignment the anisotropy of the attraction measured between jackdaws, stronger across their flight than along it[^ling2019c]. The trial flies the metric rule with its onset below any excess, so from launch it drops each of a drone's seven neighbors past 25 m, and flies 100 drones launched at spacings of 10, 15, and 25 m and measures how the polarization changes with the density and how alignment between two drones falls with the neighbor's rank. The table is written from each rule's record, and `mise render` rewrites it.

<!-- coupling -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `constant.coupling_per_s` | `0.904` | at least 0 | The rate at which a drone aligns with each neighbor, in reciprocal seconds, at a starling's scale, by default the relaxation rate inferred from the dynamics of starling flocks spread over 7 neighbors, `1 / (7 * 0.158)`, where 0.158 seconds is the geometric mean of the flocks' relaxation times |
| `anisotropic.across` | `1.0` | at least 0 | The added share of `coupling_per_s` for a neighbor directly abeam |
| `anisotropic.coupling_per_s` | `0.3` | at least 0 | The rate at which a drone aligns with a neighbor directly ahead, in reciprocal seconds |
| `metric.coupling_per_s` | `0.6` | at least 0 | The rate at which a drone aligns with each neighbor it weights, in reciprocal seconds |
| `metric.onset_k` | `5.0` |  | The excess air temperature past which the rule turns metric, in kelvin |
| `metric.range_m` | `25.0` | above 0 | The distance within which a neighbor counts while the rule is metric, in meters |
| `weighted.coupling_per_s` | `1.2` | at least 0 | The rate at which a drone would align with a neighbor at no distance, in reciprocal seconds |
| `weighted.range_m` | `20.0` | above 0 | The distance over which the rate falls by a factor of e, in meters |

<!-- /coupling -->

The chart below reads the polarization each coupling rule held at each launch spacing, the mean across four seeds beside its standard error, from the same run, and `mise measure` redraws it from a newer run.

<!-- polarization -->

The constant rule holds the polarization at 0.980 at the densest launch and 0.975 at the middle, falling to 0.947 at the sparsest, below the 0.96 of starling flocks. The anisotropic rule holds 0.9995, 0.998, and 0.986 and the distance-weighted rule 0.999, 0.996, and 0.978, whereas the metric rule, dropping every neighbor past 25 m, falls from 0.996 to 0.934 as the flock thins. Between two measured spacings the chart draws the straight line between them, which no flight measured, and three spacings over four seeds leave each rule's curve coarse.

## Noise

The default noise drives every drone's command with one amplitude, and the polarization it is measured against is the $`0.96 \pm 0.03`$ of starling flocks[^cavagna2010]. At a starling's scale the amplitude is $`10\sqrt{2T}`$ m/s per root second at 10 m/s, where the temperature $`T`$ of 0.0301 is the relaxation rate $`1 / 0.158`$ over the product of 12.13 and 17.31, the medians of the interaction range and of the coupling over the temperature inferred from starling flocks[^mora2016], which gives 2.455 and at the starling scale 0.419.

Beside that noise the default holds the zig of a starling model's waves of agitation[^hemelrijk2015], a push to one side across the heading for 0.25 s and back for 0.30 s at the model's zig force of 1 N over its bird's 80 g, followed by 1 s in which the drone starts no other. That model sets its zigs off with a predator and reproduces the wave speeds of 3.66 to 25.24 m/s measured on starling flocks under attack[^procaccini2011], whereas here a drone starts one in two ways. It zigs away from the rim once the roost's turn reaches the zig's push, after a starling's reaction time of 0.076 s, and it copies the zig of a neighbor it hears zigging, toward the side most of its zigging neighbors push, after that reaction time and the 0.05 s a starling takes to recognize a neighbor's zig, less the radio's delay. Each zigging neighbor sets the drone off within one zig with the chance one over the neighbor count, so each zig sets off one more on the mean, the critical branching ratio of an excitable network[^kinouchi2006]. Schools of fish relay startles at a branching ratio below one[^poel2022], and fish shoals dive in cascades that a model of excitable fish set off spontaneously and by their neighbors reproduces near its critical point[^gomeznava2023]. The starling model's lateral roost force turns its birds back over the roost without a zig[^hildenbrandt2010], and a starling model steering by orientation produces waves without a predator or a zig copied between birds[^hoetzlein2024]. The starling scale carries the push to 0.363 m/s², the push to the side to 8.60 s and back to 10.32 s, the refractory time to 34.4 s, the reaction time to 2.61 s, and the time to recognize a zig to 1.72 s. The candidates are a random force that decorrelates over a persistence time[^caprini2021], noise that grows with the disagreement among a drone's neighbors, the noise of a finite number of interacting animals[^jhawar2020], and noise added to the neighbors' summed heading before it is normalized, whose influence falls as the neighbors agree[^chate2008]. The trial flies 30, 100, and 200 drones and measures the polarization against the flock's size and the rate at which the flock turns of its own accord. The table is written from each rule's record, and `mise render` rewrites it.

<!-- noise -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `zig.amplitude` | `2.455` | at least 0 | The amplitude, in meters per second per root second, at a starling's scale, by default `10 * sqrt(2 * T)` at 10 meters per second, the temperature `T` of 0.0301 being the relaxation rate `1 / 0.158` over the product of 12.13 and 17.31, the medians of the interaction range and of the coupling over the temperature inferred from starling flocks |
| `zig.accel_m_s2` | `12.5` | at least 0 | The push across the heading, in meters per second squared, at a starling's scale, by default the model's zig force of 1 newton over its bird's mass of 80 grams |
| `zig.back_s` | `0.3` | above 0 | The time the drone pushes back toward its first heading, in seconds, at a starling's scale |
| `zig.branching` | `1.0` | at least 0 | The mean number of drones each zig sets off among the neighbors hearing it, by default the critical branching ratio of an excitable network |
| `zig.cue_s` | `0.05` | at least 0 | The time a drone takes to recognize a neighbor's zig, in seconds, at a starling's scale |
| `zig.reaction_s` | `0.076` | at least 0 | The time between a cue and the start of the zig, in seconds, at a starling's scale, by default the reaction time of a starling |
| `zig.refractory_s` | `1.0` | at least 0 | The time after a zig during which the drone starts none, in seconds, at a starling's scale |
| `zig.side_s` | `0.25` | above 0 | The time the drone pushes to its side, in seconds, at a starling's scale |
| `intrinsic.copying_per_s` | `1.0` | at least 0 | The rate at which a drone copies a neighbor, in reciprocal seconds |
| `intrinsic.spontaneous` | `3.7` | at least 0 | The amplitude where every neighbor agrees, in meters per second per root second |
| `persistent.persistence_s` | `1.0` | above 0 | The time over which the force decorrelates, in seconds |
| `persistent.strength_m_s2` | `3.8` | at least 0 | The root-mean-square force per axis, in meters per second squared |
| `vectorial.amplitude` | `4.1` | at least 0 | The amplitude where the neighbors fly as one, in meters per second per root second |
| `vectorial.floor` | `0.3` | above 0 | The smallest length the mean neighbor heading divides by, near the length of the mean of a neighbor count of random headings |

<!-- /noise -->

## Cohesion

The default cohesion is the starling model's[^hildenbrandt2010], a pull along the bearings of the neighbors outside the hard sphere and outside a blind angle of 90° behind the drone, summed and divided by the count of the heard neighbors outside the blind angle, those inside the hard sphere among them, and scaled by the drone's centrality, the length of the mean bearing of every sender the drone's radio holds, so a drone at the flock's edge pulls hardest and one inside it hardly at all. The model reads centrality over every bird within twice its interaction range, a set its authors call arbitrary, and the rule reads it over every sender within the radio's 50 m range that the drone still hears. At a starling's scale its hard sphere is the contact's 0.2 m and its pull at a centrality of one is the model's 1 N over 80 g, which the starling scale carries to 6.9 m and 0.36 m/s². The candidates are linear springs of nonzero rest length to each neighbor, from active elastic matter[^ferrante2013], a spring toward the neighbors' centroid, the cohesion a vision-based drone flock flies[^schilling2021], and an attraction that depends on a neighbor's distance and bearing, as fitted to pairs of fish[^calovi2018]. A control flies the default with the coupling off, since attraction alone can order a flock[^caprini2023]. The trial flies 30, 100, and 200 drones and measures the nearest-neighbor distance against the flock's size, the density at the flock's border over that at its center, the height of the first peak of the pair distribution $`g(r)`$, and the number of groups a small flock splits into. The table is written from each rule's record, and `mise render` rewrites it.

<!-- cohesion -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `centered.blind_deg` | `90.0` | at least 0 and below 360 | The angle behind the drone within which a neighbor is left out, in degrees, by default the model's 45 degrees to either side |
| `centered.sphere_m` | `0.2` | above 0 | The radius of the hard sphere within which a neighbor pulls nothing, in meters, at a starling's scale, by default half a starling's wingspan |
| `centered.weight_m_s2` | `12.5` | at least 0 | The pull of a drone at a centrality of one, in meters per second squared, at a starling's scale, by default the model's cohesion weight of 1 newton over its bird's mass of 80 grams |
| `centroid.stiffness_per_s2` | `0.1` | at least 0 | The acceleration per meter of offset from the centroid, in reciprocal square seconds |
| `elastic.rest_m` | `18.0` | above 0 | The spring's rest length, in meters |
| `elastic.stiffness_per_s2` | `0.024` | at least 0 | The acceleration per meter of stretch, in reciprocal square seconds |
| `fitted.bearing` | `0.5` | at least 0 | The share by which a neighbor ahead pulls harder and one behind pulls less |
| `fitted.range_m` | `40.0` | above 0 | The distance past which the attraction falls, in meters |
| `fitted.rest_m` | `18.0` | above 0 | The distance at which the attraction changes sign, in meters |
| `fitted.stiffness_per_s2` | `0.024` | at least 0 | The acceleration per meter past `rest_m`, in reciprocal square seconds |

<!-- /cohesion -->

## Choosing Among Candidates

A trial flies the default and each candidate for 60 s under each of three conditions over four seeds, and keeps a candidate only where no measurement leaves it worse than the default by more than the standard error of the two means across seeds and at least one leaves it better by more than that error. Beside the measurements, each composite's flight is replayed for 10 s through the drift of every composite in the trial, which gives a one-step likelihood for likelihood ranking[^torney2018] and the coefficient of each drift in a regression of the replayed accelerations on all of them, the force inference that builds a model from trajectories[^ferretti2020][^gerardos2025]. Each trial's table also carries how alignment between two drones falls with their distance, the time over which a drone's departure from the flock's heading stays correlated with itself, and the polarization of every seed, whose spread shows how far one launch strays from another. A trial reports these and chooses with none of them, since no goal for any of them has yet been set from a starling measurement.

The figures below are the means across seeds that `thermur fly cohesion_trial coupling_trial heading_trial noise_trial` prints.

| **Slot** | **Rule** | **Measured** |
|---|---|---|
| Cohesion | Centered, the default | *2.25 groups at 30 drones and 3.75 at 100, border density 0.26 of the center's at 100, $`g(r)`$ peak 2.72 times its mean* |
| Cohesion | Centroid spring | *2.50 groups at 30 and 5.50 at 100, border density 0.22, peak 0.02* |
| Cohesion | Elastic springs | *2.75 groups at 30 and 4.00 at 100, border density 1.23, peak 3.20* |
| Cohesion | Fitted attraction | *2.50 groups at 30 and 4.25 at 100, border density 0.79, peak 2.55* |
| Cohesion | Coupling off | *20.00 groups at 30, polarization 0.80 to 0.82* |
| Coupling | Constant, the default | *Polarization 0.980 at the densest launch, 0.975 at the middle, and 0.947 at the sparsest* |
| Coupling | Anisotropic | *Polarization 0.9995, 0.998, and 0.986* |
| Coupling | Metric | *Polarization 0.996, 0.972, and 0.934* |
| Coupling | Weighted by distance | *Polarization 0.999, 0.996, and 0.978* |
| Heading | Consensus, the default | *The farthest drones turning through 0.58 of a 20° turn and 0.13 of a 90° one, polarization 0.975* |
| Heading | Overdamped | *0.72 and 0.16, polarization 0.970* |
| Heading | Delayed | *0.68 and 0.15, polarization 0.972* |
| Noise | Zig, the default | *Polarization 0.977, 0.975, and 0.970 at 30, 100, and 200 drones, no spontaneous turn at 100* |
| Noise | Intrinsic | *Polarization 0.17, 0.12, and 0.11, turning 0.09 times a second at 100* |
| Noise | Persistent | *Polarization 0.19, 0.12, and 0.11, turning 0.06 times a second* |
| Noise | Vectorial | *Polarization 0.17, 0.11, and 0.10, turning 0.09 times a second* |

The default flock does not hold together at 100 drones over 60 s, splitting into 3.75 groups with its drones 14.1 m from their nearest neighbors against a launch spacing of 15 m, and it is less dense at its border than at its center, where starling flocks are denser at the border[^ballerini2008b]. Each candidate flies at the value its table gives, which the starling scale leaves as it is. The centroid spring pulls its drones to 7.0 m from their nearest neighbors at 100 drones and 11.6 m at 30 and lowers the peak of $`g(r)`$ to 0.02 times its mean, so it fails. The elastic springs raise the border density to 1.23 of the center's and the fitted attraction to 0.79, each by more than the error, with every other measurement within error of the default, so both pass.

The anisotropic and the distance-weighted rates hold the polarization nearer one than the constant rate as the flock thins, and an alignment flatter across the ranks, so both pass, whereas the metric rule loses order at the sparsest launch and fails. The farthest drones follow each turn within error of the default under either candidate heading rule, and the delayed and the overdamped rules pass on a spontaneous polarization of 0.972 and 0.970, nearer the 0.96 of starling flocks than the default's 0.975. The speed of a turn reads 4.0 m/s under the default on the 20° turn and zero on the 90° one, where in no flight did three of the five shells of drones around the turning drone turn through half of the turn at distinct times, so no fit through their times gives an outward speed. The intrinsic and the vectorial noises and the persistent force each drive the polarization to 0.10 to 0.19 and break the flock into 24 to 148 groups, so no candidate noise passes.

On every replayed flight the composite that flew it ranks first by one-step log-likelihood or ties for first. The margin over the default reaches 0.03 to 0.05 where a spring pulls toward the neighbors, 0.01 under the fitted attraction and on the flight flown with the coupling off, and 0.03 to 0.06 under a candidate coupling, since at one 0.02 s step per frame the Brownian increment dominates each step's residual. The heading rules tie, and so do the default, intrinsic, and vectorial noises, because the replay reads only the drift, whereas on the flight flown with the persistent force its own composite ranks first by 1.40.

[^ballerini2008a]: Ballerini et al. 2008. "Interaction ruling animal collective behavior depends on topological rather than metric distance: Evidence from a field study." *Proceedings of the National Academy of Sciences* 105(4): 1232–1237. <https://doi.org/10.1073/pnas.0711437105>

[^higham2001]: Higham 2001. "An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations." *SIAM Review* 43(3): 525–546. <https://doi.org/10.1137/S0036144500378302>

[^ballerini2008b]: Ballerini et al. 2008. "Empirical investigation of starling flocks: a benchmark study in collective animal behaviour." *Animal Behaviour* 76(1): 201–215. <https://doi.org/10.1016/j.anbehav.2008.02.004>

[^hildenbrandt2010]: Hildenbrandt et al. 2010. "Self-organized aerial displays of thousands of starlings: a model." *Behavioral Ecology* 21(6): 1349–1359. <https://doi.org/10.1093/beheco/arq149>

[^hertz1909]: Hertz 1909. "Über den gegenseitigen durchschnittlichen Abstand von Punkten, die mit bekannter mittlerer Dichte im Raume angeordnet sind." *Mathematische Annalen* 67(3): 387–398. <https://doi.org/10.1007/BF01450410>

[^viragh2014]: Virágh et al. 2014. "Flocking algorithm for autonomous flying robots." *Bioinspiration & Biomimetics* 9(2): 025012. <https://doi.org/10.1088/1748-3182/9/2/025012>

[^berman2021]: Berman et al. 2021. "Transport barriers to self-propelled particles in fluid flows." *Physical Review Fluids* 6(1): L012501. <https://doi.org/10.1103/PhysRevFluids.6.L012501>

[^vasarhelyi2018]: Vásárhelyi et al. 2018. "Optimized flocking of autonomous drones in confined environments." *Science Robotics* 3(20): eaat3536. <https://doi.org/10.1126/scirobotics.aat3536>

[^clausen2003]: Clausen and Jacquet 2003. "Optimized Link State Routing Protocol (OLSR)." RFC 3626. <https://doi.org/10.17487/RFC3626>

[^clausen2011]: Clausen et al. 2011. "Mobile Ad Hoc Network (MANET) Neighborhood Discovery Protocol (NHDP)." RFC 6130. <https://doi.org/10.17487/RFC6130>

[^gautrais2012]: Gautrais et al. 2012. "Deciphering Interactions in Moving Animal Groups." *PLoS Computational Biology* 8(9): e1002678. <https://doi.org/10.1371/journal.pcbi.1002678>

[^hemelrijk2011]: Hemelrijk and Hildenbrandt 2011. "Some causes of the variable shape of flocks of birds." *PLoS ONE* 6(8): e22479. <https://doi.org/10.1371/journal.pone.0022479>

[^lizarraga2026]: Lizárraga and de Aguiar 2026. "Collective turns in spinless flocks." *Physical Review E* 114(3): 035401. <https://doi.org/10.1103/6976-lcw7>

[^cavagna2015]: Cavagna et al. 2015. "Flocking and Turning: a New Model for Self-organized Collective Motion." *Journal of Statistical Physics* 158(3): 601–627. <https://doi.org/10.1007/s10955-014-1119-3>

[^geiss2022]: Geiß et al. 2022. "Signal propagation and linear response in the delay Vicsek model." *Physical Review E* 106(5): 054612. <https://doi.org/10.1103/PhysRevE.106.054612>

[^bialek2012]: Bialek et al. 2012. "Statistical mechanics for natural flocks of birds." *Proceedings of the National Academy of Sciences* 109(13): 4786–4791. <https://doi.org/10.1073/pnas.1118633109>

[^mora2016]: Mora et al. 2016. "Local equilibrium in bird flocks." *Nature Physics* 12(12): 1153–1157. <https://doi.org/10.1038/nphys3846>

[^ling2019b]: Ling et al. 2019. "Behavioural plasticity and the transition to order in jackdaw flocks." *Nature Communications* 10(1): 5174. <https://doi.org/10.1038/s41467-019-13281-4>

[^evangelista2017]: Evangelista et al. 2017. "Three-dimensional trajectories and network analyses of group behaviour within chimney swift flocks during approaches to the roost." *Proceedings of the Royal Society B: Biological Sciences* 284(1849): 20162602. <https://doi.org/10.1098/rspb.2016.2602>

[^ling2019c]: Ling et al. 2019. "Local interactions and their group-level consequences in flocking jackdaws." *Proceedings of the Royal Society B: Biological Sciences* 286(1906): 20190865. <https://doi.org/10.1098/rspb.2019.0865>

[^cavagna2010]: Cavagna et al. 2010. "Scale-free correlations in starling flocks." *Proceedings of the National Academy of Sciences* 107(26): 11865–11870. <https://doi.org/10.1073/pnas.1005766107>

[^hemelrijk2015]: Hemelrijk et al. 2015. "What underlies waves of agitation in starling flocks." *Behavioral Ecology and Sociobiology* 69(5): 755–764. <https://doi.org/10.1007/s00265-015-1891-3>

[^procaccini2011]: Procaccini et al. 2011. "Propagating waves in starling, *Sturnus vulgaris*, flocks under predation." *Animal Behaviour* 82(4): 759–765. <https://doi.org/10.1016/j.anbehav.2011.07.006>

[^kinouchi2006]: Kinouchi and Copelli 2006. "Optimal dynamical range of excitable networks at criticality." *Nature Physics* 2(5): 348–351. <https://doi.org/10.1038/nphys289>

[^poel2022]: Poel et al. 2022. "Subcritical escape waves in schooling fish." *Science Advances* 8(25): eabm6385. <https://doi.org/10.1126/sciadv.abm6385>

[^gomeznava2023]: Gómez-Nava et al. 2023. "Fish shoals resemble a stochastic excitable system driven by environmental perturbations." *Nature Physics* 19(5): 663–669. <https://doi.org/10.1038/s41567-022-01916-1>

[^hoetzlein2024]: Hoetzlein 2024. "Flock2: A model for orientation-based social flocking." *Journal of Theoretical Biology* 593: 111880. <https://doi.org/10.1016/j.jtbi.2024.111880>

[^caprini2021]: Caprini and Marini Bettolo Marconi 2021. "Inertial self-propelled particles." *The Journal of Chemical Physics* 154(2): 024902. <https://doi.org/10.1063/5.0030940>

[^jhawar2020]: Jhawar et al. 2020. "Noise-induced schooling of fish." *Nature Physics* 16(4): 488–493. <https://doi.org/10.1038/s41567-020-0787-y>

[^chate2008]: Chaté et al. 2008. "Collective motion of self-propelled particles interacting without cohesion." *Physical Review E* 77(4): 046113. <https://doi.org/10.1103/PhysRevE.77.046113>

[^ferrante2013]: Ferrante et al. 2013. "Elasticity-Based Mechanism for the Collective Motion of Self-Propelled Particles with Springlike Interactions: A Model System for Natural and Artificial Swarms." *Physical Review Letters* 111(26): 268302. <https://doi.org/10.1103/PhysRevLett.111.268302>

[^schilling2021]: Schilling et al. 2021. "Vision-Based Drone Flocking in Outdoor Environments." *IEEE Robotics and Automation Letters* 6(2): 2954–2961. <https://doi.org/10.1109/LRA.2021.3062298>

[^calovi2018]: Calovi et al. 2018. "Disentangling and modeling interactions in fish with burst-and-coast swimming reveal distinct alignment and attraction behaviors." *PLOS Computational Biology* 14(1): e1005933. <https://doi.org/10.1371/journal.pcbi.1005933>

[^caprini2023]: Caprini and Löwen 2023. "Flocking without Alignment Interactions in Attractive Active Brownian Particles." *Physical Review Letters* 130(14): 148202. <https://doi.org/10.1103/PhysRevLett.130.148202>

[^torney2018]: Torney et al. 2018. "Inferring the rules of social interaction in migrating caribou." *Philosophical Transactions of the Royal Society B: Biological Sciences* 373(1746): 20170385. <https://doi.org/10.1098/rstb.2017.0385>

[^ferretti2020]: Ferretti et al. 2020. "Building General Langevin Models from Discrete Datasets." *Physical Review X* 10(3): 031018. <https://doi.org/10.1103/PhysRevX.10.031018>

[^gerardos2025]: Gerardos and Ronceray 2025. "Principled Model Selection for Stochastic Dynamics." *Physical Review Letters* 135(16): 167401. <https://doi.org/10.1103/ltdt-hvh7>
