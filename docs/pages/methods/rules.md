# The Interaction Rules

Each drone in *Thermur* steers by the sum of a few rules, and each rule reads only what that one drone senses. The controller has a slot for each part of the motion, meaning the heading a drone turns toward, the rate it aligns with each neighbor at, the random increments driving it, the force holding it at a spacing, the contact keeping two drones apart, the roost holding the flock over one place, and the speed it flies at. Each slot holds a rule borrowed from another physical system, a starling flock, a school of fish, or a swarm of active particles. An open slot starts from a default and flies each candidate rule against it, keeping a candidate only where the measurements favor it. In the flights below only the anisotropic and the distance-weighted rates for the coupling and the overdamped limit for the heading pass that test, and every slot still holds its default.

## The Neighbors a Drone Senses

A drone senses its seven nearest neighbors by rank rather than within a fixed range, the topological interaction measured in starling flocks[^ballerini2008a]. For each neighbor it reads the offset, the distance, and the velocity as its radio last heard them, together with the velocity a neighbor held half a second before that for the rule that reads a delay, and it reads its own position, its own command, and the excess temperature and the wind of the air at its position. A rule receives that view and nothing else.

By rank a drone senses the seven nearest of the senders its radio holds, fewer only where it holds fewer than seven, whereas the number within a fixed range swings as a flock gathers and splits. The drawing below flies the front page's flock under a TypeScript copy of the default composite, whose step and radio the site's suite checks against the package's own from the same states and draws. The pointer acts as a falcon through a push the drawing adds beside the composite, whose default slots read no air. The marked bird's lines run to the nearest seven of the senders its radio holds, ranked by where the radio last heard them, and the upper strip counts them, whereas the ring counts the birds within the fixed range by the distance between the drones the drawing flies.

<!-- neighbors -->

The default cohesion reads one quantity past the seven, the drone's centrality, over every sender its radio holds. The suite checks this with a property test, which moves a drone that another drone cannot hear to a random place still out of that drone's radio range and asserts that the second drone's acceleration and the amplitude of its noise stay the same, under the default composite and under every candidate.

## The Flight

A flight integrates each drone's command and spin under the sum of the slots, and its velocity and position under the plant and the wind below, through the Euler-Maruyama scheme, one step of diffrax's Euler solver at a time, its noise the Brownian increment a virtual Brownian tree draws, which carries the $`\sqrt{\Delta t}`$ scaling of a stochastic differential equation[^higham2001]. Every draw derives from one seed, and several seeds fly together as one batch. The flock launches in a thin horizontal cloud, its axes in the proportion 5.6 to 2.8 to 1 measured on starling flocks, its thin axis vertical[^ballerini2008b]. After each step a drone keeps as its speed the part of its new command along its old one, since a turn taken in one explicit step otherwise adds speed in proportion to the square of the step, and under a heading rule carrying a spin its command turns once more by the change in its spin across the step, which keeps a spin with no friction from gaining energy step by step.

The table is written from the record each flight reads, set under `[tool.thermur.plan]` or through a `--plan.` flag of `thermur fly`, and `mise render` rewrites it.

<!-- plan -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `airspeeds_m_s` | `[6.9, 15.2]` |  | The slowest and the fastest speed at which a drone's speed stays stable under one step of the integrator, in meters per second, the range the centers of starling flocks fly at |
| `axes` | `[5.6, 2.8, 1.0]` |  | The launch cloud's length, width, and depth, in proportion, the thin horizontal shape measured on starling flocks, whose long axis holds no fixed bearing to their motion, launched with its long axis along the course and its thin axis vertical |
| `center_m` | `[1000.0, 1000.0, 100.0]` |  | The center of the launch cloud, east, north, and above the ground, in meters, by default on the parametric threat's course to the front it crosses within two minutes, its height the lowest the launch's center takes, which `clearance_sd` raises for a deeper cloud |
| `clearance_sd` | `4.0` | above 0 | The vertical standard deviations of the launch cloud its center sits at least above the ground, by default 4, past which a Gaussian draw falls for 0.003% of drones, so about one flock of 1,000 in 32 sets a drone on the ground |
| `course_deg` | `180.0` |  | The launch heading in degrees counterclockwise from east, by default west |
| `cruise_m_s` | `10.0` | above 0 | The speed each drone launches and cruises at, in meters per second |
| `drones` | `1000` | at least 1 | The size of the flock, by default near the median of the ten starling flocks of 448 to 2,631 birds the topological interaction was measured on |
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

Each default the slots hold is a value a starling study states at a starling's scale, and a flight carries every one of them to the drones' scale through one factor. The drone flock is taken as the starling flock magnified at the cruise speed of 10 m/s the starling model flies[^hildenbrandt2010], by the launch's nearest-neighbor distance over the 0.69 m that model's flock of 781 birds holds. The launch's nearest-neighbor distance is the mean distance points at random hold at a given density[^hertz1909], 0.554 times the cube root of the volume per point, averaged over the density of the Gaussian cloud the plan launches, which at a launch spacing of 15 m gives 23.7 m and a factor of 34.4. At one speed a time scales by the same factor as a length, so a slot multiplies each length by the factor, divides each acceleration and each rate by it, divides a spring's stiffness by its square, and divides a noise's amplitude by its square root, which keeps the spread the noise drives in proportion to the spacing. `Composite.from_settings` scales every slot, so a value set under `[tool.thermur]` or through a flag reads at a starling's scale.

## The Plant and the Wind

A drone's autopilot carries its velocity through the air toward its command over a first-order lag of one second[^viragh2014], holding the command within the speeds and the response within the accelerations the PX4 multicopter defaults set, with the horizontal speed raised within PX4's range to the 15.2 m/s of the fastest starling flock measured[^ballerini2008b], and the descent's speed and acceleration raised within PX4's ranges to match the climb's. A bound tighter below than above turns noise of zero mean on a vertical command into a mean climb, which under PX4's own descent bounds lifts a flight of 100 drones to a mean height of 160 m over two minutes against the roost's 100 m. The plant holds no gravity term, since the autopilot carries the drone's weight, and it holds the drone's own bounds, so the starling scale leaves it as it is. A drone moves over the ground at its velocity through the air plus the wind at its position[^berman2021], so the WRF-SFIRE winds carry the flock in place of a drag force, whereas the parametric threat's air is still. The table is written from the plant's record, set under `[tool.thermur.plant]` or through a `--plant.` flag, and `mise render` rewrites it.

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

A drone hears its neighbors only through its radio, which broadcasts each drone's state five times a second and delivers it one second late, the sensor update period and the delay of the setup the outdoor drone flock was simulated under[^vasarhelyi2018], whose update period matches the earlier quadcopter flock's[^viragh2014]. A packet arrives only where its sender sat within 50 m of the drone as it was sent, the distance past which the outdoor flock's radios began to lose long runs of packets, nearly every packet arriving from a peer within 20 m. Each drone holds the 16 nearest senders it hears in a table whose size stays the same at any flock size. It learns a sender beyond the table from the neighbor list each member of the table broadcasts, as the Optimized Link State Routing protocol (OLSR) fills its set of 2-hop neighbors, the nodes its own neighbors hear, in section 8.2 of its specification[^clausen2003], and from the 8 drones before it and the 7 after it along a Z-order curve, which orders the broadcasts by interleaving the bits of their coordinates so that drones near one another mostly sit near one another. The radio is a drone's own, so the starling scale leaves its delay, its rate, its loss, and its range as they are. A drone ranks its seven nearest neighbors by the positions it heard, as a starling interacts with a fixed number of the neighbors it perceives[^ballerini2008a], so a neighbor heard from within range may since have flown past it, and a drone that has heard fewer than seven others reads each missing neighbor as one sitting on the drone at its own command. Each packet from each sender is lost with a chance of 0.4, the middle of the 20 to 60% the outdoor flock's radios lost, and a drone keeps the last state it heard from a sender until another packet arrives. A drone hears a sender for three packet periods after its last packet, 0.6 s, and then forgets it, the neighbor hold time of three refresh intervals that the routing protocols of mobile ad hoc networks set[^clausen2003][^clausen2011]. Each packet carries the sender's position, its velocity, and the state of its noise slot. The replay that scores each composite's drift delivers every packet sent from within range, keeping the delay. The settings table below is written from the radio's record, and `mise render` rewrites it.

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

The roost slot keeps the flock flat through the vertical part of the starling model's roost term[^hildenbrandt2010], a pull toward the roost's height in proportion to the height away, and against the drone's climb over the ground at twice the square root of that pull's stiffness, which damps the height critically, since the starling model tunes its speed's relaxation with that pull so its flocks fly level when nothing turns them, through a lift that rises with speed, and the drone's plant carries no such lift. That model's vertical weight over its bird's mass, 0.2 N per meter over 80 g, reads per meter because the model multiplies it by the height away, although its table of parameters lists it in newtons, and the starling scale carries its 2.5 per square second to 0.0021. The height defaults to the 100 m of the launch's center, which a flock of more than 304 drones launches above. The slot also holds the flock over a disk of 1000 m radius around the launch's center, at (1000, 1000) m inside the parametric threat's 4 km by 2 km grid, through the turn fish in a circular tank take from its wall[^gautrais2012]. A drone heading for the rim turns across its heading, away from the rim, at a rate of 0.94 over its time to reach the rim at its ground speed, the strength fitted to the fish, so the turn grows as the rim nears. The slot applies that rate as an acceleration across the heading at the ground speed, capped at the plant's horizontal bound of 5 m/s², whereas the fish relax their turning rate toward it, and a drone outside the disk turns at the cap. The strength is a ratio of times, so the starling scale leaves it as it is, and the radius is set at the drone's scale. The table is written from the slot's record, and `mise render` rewrites it.

<!-- roost -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `bound_m_s2` | `5.0` | above 0 | The largest turning acceleration, in meters per second squared, by default the plant's bound on a horizontal acceleration |
| `center_m` | `[1000.0, 1000.0]` |  | The disk's center east and north, in meters, by default the launch's |
| `height_m` | `100.0` |  | The height the drone is pulled toward, in meters above the ground, by default that of the launch's center, which a flock of more than 304 drones launches above |
| `radius_m` | `1000.0` | above 0 | The disk's radius, in meters |
| `stiffness_per_s2` | `2.5` | at least 0 | The acceleration toward the height per meter away from it, in reciprocal square seconds, at a starling's scale, by default the starling model's vertical weight of 0.2 newtons per meter over its bird's mass of 80 grams, read per meter since the model multiplies it by the height away, although its table of parameters lists it in newtons |
| `turn` | `0.94` | at least 0 | The turn rate times the time to reach the rim, in radians, by default the strength fitted to fish turning from the wall of their tank |

<!-- /roost -->

## The Speed

The default speed rule is the starling model's speed control[^hildenbrandt2010], which relaxes a drone's speed along its command toward the cruise speed, so a drone left behind in a turn catches up and one pushed ahead falls back, over 1 s toward 10 m/s at a starling's scale, the model's own values, which a later study of that model flies as well[^hemelrijk2011], and which the starling scale carries to 34.4 s. It drives no random increment along the command, so the speed changes only through the relaxation and the plant's bounds. A setting holds every drone at the cruise speed instead, the control under which no speed fluctuates[^cavagna2010], removing the part of every acceleration along the command, and the settings refuse a speed rule whose stiffness at any speed between 6.9 and 15.2 m/s, the range of the mean speeds measured on starling flocks[^ballerini2008b], times the step exceeds two, past which one step widens the gap between two nearly equal speeds there rather than narrowing it. The candidate gives the speed a channel of its own. A drone imitates its neighbors' speeds at the rates the coupling sets, accelerating along its command by the weighted differences between their speeds and its own[^bialek2014], and a marginal potential $`V = \lambda (\lVert v \rVert^2 - v_0^2)^4 / v_0^6`$ holds it about the cruise speed $`v_0`$, its force the gradient of $`-V`$ with respect to the command, whose curvature at $`v_0`$ is zero, so the confinement leaves a small departure from the cruise speed free and restores a large one[^cavagna2022]. A random increment along the command drives the speed at an amplitude of its own, and the candidate's values are set at the drones' scale, so the starling scale leaves them as they are. At its defaults one step stays within the same bound at every speed from 6.9 to 15.2 m/s. A control flies the default holding every drone at the cruise speed. The trial flies 30, 100, and 200 drones and measures the polarization against the flock's size. The table is written from each rule's record, set under `[tool.thermur.speed]` or through a `--speed.` flag of `thermur fly`, and `mise render` rewrites it.

<!-- speed -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `cruise.fixed` | `False` |  | Whether a flight holds each drone at the cruise speed, leaving no speed to fluctuate |
| `cruise.cruise_m_s` | `10.0` | above 0 | The speed the drone relaxes toward, in meters per second, by default the model's cruise speed of 10 meters per second, which the plan launches the flock at |
| `cruise.relaxation_s` | `1.0` | above 0 | The time over which the drone's speed relaxes toward `cruise_m_s`, in seconds, at a starling's scale, by default the model's speed control of 1 second |
| `marginal.fixed` | `False` |  | Whether a flight holds each drone at the cruise speed, leaving no speed to fluctuate |
| `marginal.amplitude` | `1.0` | at least 0 | The amplitude of the random increments of the speed, in meters per second per root second, at a drone flock's scale |
| `marginal.confinement_per_s` | `0.2` | above 0 | The strength of the potential, in reciprocal seconds, at a drone flock's scale |

<!-- /speed -->

Under the defaults, 1000 drones over four seeds keep their nearest neighbors 11.4 m away at 120 s and 12.2 m at 300 s on the mean, below the 12 to 30 m the outdoor flock kept[^vasarhelyi2018] at 120 s and inside it at 300 s, at a polarization of 0.71 to 0.74 and 0.77 to 0.80 across the seeds, below the 0.96 of starling flocks. Launched about a center 149 m up, four times its cloud's vertical deviation of 37.2 m, the settled flock's mean height is 112 m and 104 m, single drones ranging from 17 m to 247 m at 120 s and from 21 m to 251 m by 300 s, so no kept position through 300 s sits under the ground, whereas a drone sits outside the roost's disk in 5 of them through 120 s and 7 through 300 s. A drone is zigging, waiting to, or in its refractory time in 24% and 20% of the frames. A flock's shape is averaged over time, as the starling model's flocks are, each property measured once per second over 30 minutes[^hemelrijk2011]. Read each second past the first 20 s, the largest group the flock splits into stands at 1 to 4.05 to 10.9 through 120 s and 1 to 2.97 to 10.4 through 300 s on the mean over the seeds, its longest axis running 10.2 to 11.5 and 9.52 to 10.9 times its thinnest across the seeds, well past the span of the starling flocks the starling model is checked against[^hildenbrandt2010], whose longest axes run 4.02 to 6.94 times their thinnest, while through 300 s its middle axis stands at 2.77 to 3.08 times its thinnest, inside those flocks' 1.90 to 3.58. The shape check compares the longest axis alone, because the longest axis over the thinnest does not depend on flock size in the flocks over Rome behind that span, of 448 to 2631 birds, nor in the starling model's flocks matched to them, whereas the model's middle axis over the thinnest rises with flock size, a dependence the study lists among the model's mismatches with the flocks over Rome[^hildenbrandt2010]. Counting two drones in one group where a chain of hops under 45 m joins them, the flock flies in 21, 23, 21, and 21 groups across the seeds at 120 s and 18, 16, 18, and 15 at 300 s, and read each second through 300 s its largest group holds no fewer than 445 to 650 of the drones.

The tests of a flight fly 100 drones, whose warm flight takes a quarter of the CPU time per simulated second that 1000 drones take. There the drones keep their nearest neighbors 16.4 m away at 120 s and 15.0 m at 300 s, inside the outdoor flock's 12 to 30 m, at a polarization of 0.72 to 0.80 and 0.80 to 0.81. Every drone stays over the roost's disk at every frame, the settled flock's mean height is 99.5 m at both lengths, single drones ranging from 29 m to 150 m at 120 s and from 29 m to 159 m by 300 s, and a drone is zigging, waiting to, or in its refractory time in 23% and 18% of the frames. The largest group stands at 1 to 2.23 to 6.79 through 120 s and 1 to 1.82 to 5.90 through 300 s on the mean over the seeds, so on the mean over four seeds its longest axis falls inside the starling flocks' span. Through 120 s that mean sits at the span's edge, since the four seeds read 4.61 to 10.3 and twelve seeds 6.67 on the mean, whose 95% interval of 5.55 to 7.78 straddles 6.94, so four seeds cannot show that the 120 s shape stays inside the span. The flock flies in 10, 8, 8, and 4 groups across the seeds at 120 s and 9, 6, 8, and 4 at 300 s, its largest group holding 39 to 58 of the drones at 300 s.

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

The default heading rule is a spinless second-order consensus[^lizarraga2026], which relaxes a drone's velocity toward each neighbor's at the rate the coupling sets and toward the neighbors' mean as well, where the source relaxes toward the whole flock's, and leaves the speed to the speed slot, the cohesion, the contact, the noise, and the roost turning the command only across it. Its pull toward the mean is the source's aggregation strength over its alignment strength per neighbor, 0.01 to 50, times the coupling's rate, so it carries to the drones' scale as the coupling does. It holds no state of its own between steps, so a step costs no more than the pull it sums.

The candidates are the inertial spin model[^cavagna2015], its overdamped limit, two variants of the spin, and alignment on the neighbors' velocities half a second old[^geiss2022]. Under the inertial spin the pull of a drone's neighbors exerts a torque on the drone's spin, the spin carries an inertia $`\chi`$ and a friction $`\eta`$, and the spin turns the command, so a turn crosses the flock as a wave at the neighbor spacing $`a`$ times $`\sqrt{J n_c / \chi}`$ rather than spreading by diffusion, $`J n_c`$ being the sum of the rates the coupling sets. Its inertia of 0.0112 s at a starling's scale, 0.386 s at the drones', puts that speed near 30 m/s across the launch, the middle of the 20 to 40 m/s measured on turning starling flocks[^attanasi2014], with $`a`$ the root-mean-square distance to a drone's seven neighbors, 47, 44, and 43 m at the 30, 100, and 200 drones the noise and cohesion trials fly. A turn reaches across a flock of size $`L`$ where $`\eta < a \sqrt{J n_c \chi} / L`$, which with $`L`$ the largest distance between two drones at launch, 256, 493, and 659 m, reads 0.049, 0.024, and 0.017, whereas with the radio's delay and the plant's lag in the loop the spin holds its order only above a friction of 0.069 to 0.091 across those sizes. The friction of 0.3 therefore sits above the bound at every size, and in the trial below the spin carries neither turn to the farthest drones. The cohesion, the contact, and the noise's force turn the command directly, as the force Carruitero et al. add to the velocity's own equation[^carruitero2024], rather than as a torque on the spin, under which a drone's offset from its neighbors answers the cohesion only through the spin's inertia and the flock comes apart. The noise reaches the spin rather than the heading, scaled by $`\sqrt{\eta} / \chi`$, which leaves the overdamped limit's noise at a friction of one as the inertia falls to zero, that limit being a drone's command relaxing toward its neighbors' with no spin carried between steps and its speed held. One variant of the spin adds to the alignment a quartic term in the difference between two velocities, weak under the small differences of spontaneous fluctuation and strong under a turn, so the turn travels while the spontaneous fluctuations stay damped[^cavagna2026]. The other weights a neighbor ahead of a drone above one behind, so the torques two drones exert on each other differ[^dadhichi2020]. The trial turns one drone through 20° and through 90° and measures how far the turn reaches across the flock and how fast it crosses, each drone's rotation read against the same drone in a flight from the same launch and the same noise with no drone turned, so a flock rotating as a whole reads no reach. The reach is favored nearest the whole turn, since a turn crosses a starling flock linearly in time with negligible attenuation[^attanasi2014]. Beside these the trial measures the polarization and the frequency at which the spontaneous velocity correlation $`C(k,\omega)`$ peaks once each mode's mean over the flight is removed, favored nearest zero, where the overdamped Lorentzian measured on starling flocks peaks[^cavagna2026], so a rule whose spontaneous fluctuations travel as spin waves scores below one whose fluctuations relax. The table is written from each rule's record, the overdamped limit declaring no parameter, and `mise render` rewrites it.

<!-- heading -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `consensus.centering` | `0.0002` | at least 0 | The rate at which a drone's velocity relaxes toward its neighbors' mean, as a share of the coupling's mean rate per neighbor, by default the source's aggregation strength of 0.01 over its alignment strength of 50 per neighbor, so it carries to any scale the coupling does |
| `delayed.lag_s` | `0.5` | above 0 | The age of the neighbor states the slot reads, in seconds |
| `inertialspin.friction` | `0.3` | above 0 | The friction on the spin, which relaxes it at `friction / inertia_s` |
| `inertialspin.inertia_s` | `0.0112` | above 0 | The inertia of the spin, in seconds, at a starling's scale, by default the one carrying a turn across the launch at 30 meters per second, the middle of the 20 to 40 measured on turning starling flocks, `7 * 0.904 * 1.264**2 / 30**2`, where 1.264 meters is the launch's root-mean-square distance to its seven neighbors at a starling's scale |
| `nonmutual.friction` | `0.3` | above 0 | The friction on the spin, which relaxes it at `friction / inertia_s` |
| `nonmutual.inertia_s` | `0.0112` | above 0 | The inertia of the spin, in seconds, at a starling's scale, by default the one carrying a turn across the launch at 30 meters per second, the middle of the 20 to 40 measured on turning starling flocks, `7 * 0.904 * 1.264**2 / 30**2`, where 1.264 meters is the launch's root-mean-square distance to its seven neighbors at a starling's scale |
| `nonmutual.asymmetry` | `0.5` | at least 0 and at most 1 | The added share of a neighbor's weight per unit cosine of its bearing off the drone's velocity, which takes the weight of a neighbor straight behind to zero at one |
| `quartic.friction` | `0.6` | above 0 | The friction on the spin, which relaxes it at `friction / inertia_s` |
| `quartic.inertia_s` | `0.0112` | above 0 | The inertia of the spin, in seconds, at a starling's scale, by default the one carrying a turn across the launch at 30 meters per second, the middle of the 20 to 40 measured on turning starling flocks, `7 * 0.904 * 1.264**2 / 30**2`, where 1.264 meters is the launch's root-mean-square distance to its seven neighbors at a starling's scale |
| `quartic.quartic` | `10.0` | at least 0 | The strength of the quartic term over the quadratic one, at a difference in velocity equal to the cruise speed |

<!-- /heading -->

The chart below reads the share of the first drone's turn that the farthest of five shells of drones turned through under each heading rule, beyond the same drones flown with no turn, the mean across four seeds beside its standard error, from a run of `thermur fly coupling_trial heading_trial` that `mise measure` wrote out, and `mise measure` redraws it from a newer run.

<!-- reach -->

Under every heading rule the farthest drones end the flight turned through under a thirtieth of a 20° turn and under a fortieth of a 90° one beyond the same drones flown with no turn, the default reaching 0.003 and 0.002, the delayed rule 0.003 and 0.002, the nonmutual variant 0.025 and 0.003, the quartic variant 0.032 and 0.023, and the inertial spin 0.026 and 0.006. The chart leaves out the overdamped limit, which reaches 0.003 and 0.002. On three of the four seeds the farthest drones end within a hundredth of a 20° turn of their unturned flight, so one seed carries most of each mean and the standard errors run from 57% to 87% of the means. Four seeds still leave the inertial spin and both variants of the spin carrying the 20° turn farther than the default by more than the error, and the quartic variant the 90° one as well, whereas the delayed rule and the overdamped limit stay within error of the default on both.

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

The constant rule holds the polarization at 0.980 at the densest launch and 0.973 at the middle, falling to 0.948 at the sparsest, below the 0.96 of starling flocks. The anisotropic rule holds 0.9993, 0.997, and 0.987 and the distance-weighted rule 0.999, 0.996, and 0.981, whereas the metric rule, dropping every neighbor past 25 m, falls from 0.994 to 0.935 as the flock thins. Between two measured spacings the chart draws the straight line between them, which no flight measured, and three spacings over four seeds leave each rule's curve coarse.

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

The default cohesion is the starling model's[^hildenbrandt2010], a pull along the bearings of the neighbors outside the hard sphere and outside a blind angle of 90° behind the drone, summed and divided by the count of the heard neighbors outside the blind angle, those inside the hard sphere among them, and scaled by the drone's centrality, the length of the mean bearing of every sender the drone's radio holds, so a drone at the flock's edge pulls hardest and one inside it hardly at all. The model reads centrality over every bird within twice its interaction range, a set its authors call arbitrary, and the rule reads it over the senders the drone's radio holds, at most the 16 nearest it still hears within the radio's 50 m range. At a starling's scale its hard sphere is the contact's 0.2 m and its pull at a centrality of one is the model's 1 N over 80 g, which the starling scale carries to 6.9 m and 0.36 m/s². The candidates are linear springs of nonzero rest length to each neighbor, from active elastic matter[^ferrante2013], a spring toward the neighbors' centroid, the cohesion a vision-based drone flock flies[^schilling2021], and an attraction that depends on a neighbor's distance and bearing, as fitted to pairs of fish[^calovi2018]. A control flies the default with the coupling off, since attraction alone can order a flock[^caprini2023]. The trial flies 30, 100, and 200 drones and measures the nearest-neighbor distance against the flock's size, the density at the flock's border over that at its center, the height of the first peak of the pair distribution $`g(r)`$, and the number of groups a small flock splits into. The table is written from each rule's record, and `mise render` rewrites it.

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

The figures below are the means across seeds that `thermur fly cohesion_trial coupling_trial heading_trial noise_trial speed_trial` prints.

| **Slot** | **Rule** | **Measured** |
|---|---|---|
| Cohesion | Centered, the default | *2.25 groups at 30 drones and 4.25 at 100, border density 2.36 of the center's at 100, $`g(r)`$ peak 2.13 times its mean* |
| Cohesion | Centroid spring | *3.00 groups at 30 and 4.00 at 100, border density 0.14, peak 0.70* |
| Cohesion | Elastic springs | *3.00 groups at 30 and 3.50 at 100, border density 0.59, peak 2.92* |
| Cohesion | Fitted attraction | *2.50 groups at 30 and 3.75 at 100, border density 0.35, peak 2.44* |
| Cohesion | Coupling off | *16.50 groups at 30, polarization 0.82 to 0.83* |
| Coupling | Constant, the default | *Polarization 0.980 at the densest launch, 0.973 at the middle, and 0.948 at the sparsest* |
| Coupling | Anisotropic | *Polarization 0.9993, 0.997, and 0.987* |
| Coupling | Metric | *Polarization 0.994, 0.976, and 0.935* |
| Coupling | Weighted by distance | *Polarization 0.999, 0.996, and 0.981* |
| Heading | Consensus, the default | *The farthest drones turning through 0.003 of a 20° turn and 0.002 of a 90° one beyond the same drones flown with no turn, polarization 0.973* |
| Heading | Delayed | *0.003 and 0.002, polarization 0.973* |
| Heading | Inertial spin | *0.026 and 0.006, polarization 0.982* |
| Heading | Nonmutual | *0.025 and 0.003, polarization 0.979* |
| Heading | Overdamped | *0.003 and 0.002, polarization 0.971* |
| Heading | Quartic | *0.032 and 0.023, polarization 0.984* |
| Noise | Zig, the default | *Polarization 0.981, 0.973, and 0.972 at 30, 100, and 200 drones, no spontaneous turn at 100* |
| Noise | Intrinsic | *Polarization 0.18, 0.09, and 0.06, no spontaneous turn at 100* |
| Noise | Persistent | *Polarization 0.17, 0.09, and 0.07, turning 0.01 times a second at 100* |
| Noise | Vectorial | *Polarization 0.19, 0.10, and 0.07, turning 0.01 times a second* |
| Speed | Cruise, the default | *Polarization 0.981, 0.973, and 0.972 at 30, 100, and 200 drones* |
| Speed | Marginal | *Polarization 0.977, 0.975, and 0.971* |
| Speed | Held at the cruise speed | *Polarization 0.980, 0.971, and 0.970* |

The default flock does not hold together at 100 drones over 60 s, splitting into 4.25 groups with its drones 13.9 m from their nearest neighbors against a launch spacing of 15 m, and at 200 drones it is less dense at its border than at its center, 0.06 of the center's, where starling flocks are denser at the border[^ballerini2008b], whereas at 100 its border reads 2.36 times its center with a standard error of 0.79. A trial scales each candidate from a starling's scale as `Composite.from_settings` scales the default, which carries the inertial spin's inertia to the drones' scale and leaves every other candidate at the value its table gives. Every candidate cohesion lowers the border density at 100 drones to between 0.14 and 0.59 of the center's, each by more than the error, so none passes, although the elastic springs raise the peak of $`g(r)`$ to 2.92 times its mean, by more than the error.

The anisotropic and the distance-weighted rates hold the polarization nearer one than the constant rate as the flock thins, and an alignment flatter across the ranks, so both pass, each gain 3.8 to 5.8 times the combined standard error of the two means, whereas the metric rule loses order at the sparsest launch and fails. Of the heading rules only the overdamped limit passes, holding a spontaneous polarization of 0.971, nearer the 0.96 of starling flocks than the default's 0.973, while every other measurement leaves it within error of the default. Its polarization sits 0.0021 nearer 0.96 than the default's against a combined standard error of 0.0021, clearing it by 0.00004, so more seeds could overturn the pass. The inertial spin and both variants carry the 20° turn farther than the default by more than the error, whereas each holds a spontaneous polarization farther from 0.96, 0.979 to 0.984, so each fails, and the delayed rule stays within error of the default on every measurement. No rule carries either turn across the flock. The speed of the turn reads its fallback of zero under every rule, since in no flight did three of the five shells of drones around the turning drone turn through half of the turn at distinct times, so no fit through their times gives an outward speed, and the speed no longer separates the rules. The spontaneous correlation of every rule peaks at 0.025 Hz, the lowest frequency the 40 s each settled flight spans resolves, where fluctuations relaxing with no wave peak, so the frequency separates no rule either and would only mark one growing a spin wave. The trial cannot show a turn that crosses part of the flock and fades, since the reach reads the farthest shell at the flight's end, 35 s after the turn, nor a collective turn, since the turned drone holds its command and flies off from its neighbors. The intrinsic and the vectorial noises and the persistent force each drive the polarization to 0.06 to 0.19 and break the flock into 4 to 21 groups, so no candidate noise passes. The marginal speed holds the polarization at 100 drones at 0.975, farther from the 0.96 of starling flocks than the cruise control's 0.973 by more than the error and within error of it at 30 and 200 drones, so it fails.

On every replayed flight the composite that flew it ranks first by one-step log-likelihood or ties for first. The margin over the default reaches 0.03 to 0.05 where a spring pulls toward the neighbors, 0.01 under the fitted attraction and on the flight flown with the coupling off, and 0.03 to 0.07 under a candidate coupling, since at one 0.02 s step per frame the Brownian increment dominates each step's residual. On a flight under the inertial spin or either variant, whose spin the replay reads, the three forms of the spin rank first by 2.9 to 3.5 over every spinless rule, and among themselves they tie, as the spinless heading rules do on their own flights. The default, intrinsic, and vectorial noises tie, and so do the speed rules, because the replay reads only the drift, whereas on the flight flown with the persistent force its own composite ranks first by 2.12.

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

[^cavagna2010]: Cavagna et al. 2010. "Scale-free correlations in starling flocks." *Proceedings of the National Academy of Sciences* 107(26): 11865–11870. <https://doi.org/10.1073/pnas.1005766107>

[^bialek2014]: Bialek et al. 2014. "Social interactions dominate speed control in poising natural flocks near criticality." *Proceedings of the National Academy of Sciences* 111(20): 7212–7217. <https://doi.org/10.1073/pnas.1324045111>

[^cavagna2022]: Cavagna et al. 2022. "Marginal speed confinement resolves the conflict between correlation and control in collective behaviour." *Nature Communications* 13(1): 2315. <https://doi.org/10.1038/s41467-022-29883-4>

[^lizarraga2026]: Lizárraga and de Aguiar 2026. "Collective turns in spinless flocks." *Physical Review E* 114(3): 035401. <https://doi.org/10.1103/6976-lcw7>

[^cavagna2015]: Cavagna et al. 2015. "Flocking and Turning: a New Model for Self-organized Collective Motion." *Journal of Statistical Physics* 158(3): 601–627. <https://doi.org/10.1007/s10955-014-1119-3>

[^geiss2022]: Geiß et al. 2022. "Signal propagation and linear response in the delay Vicsek model." *Physical Review E* 106(5): 054612. <https://doi.org/10.1103/PhysRevE.106.054612>

[^attanasi2014]: Attanasi et al. 2014. "Information transfer and behavioural inertia in starling flocks." *Nature Physics* 10(9): 691–696. <https://doi.org/10.1038/nphys3035>

[^carruitero2024]: Carruitero et al. 2024. "Inertial spin model of flocking with position-dependent forces." *Physical Review E* 110(1): 014408. <https://doi.org/10.1103/PhysRevE.110.014408>

[^cavagna2026]: Cavagna et al. 2026. "Spin Waves without Spin Waves: A Case for Soliton Propagation in Starling Flocks." *Physical Review Letters* 137(9): 098301. <https://doi.org/10.1103/sppm-qsnb>

[^dadhichi2020]: Dadhichi et al. 2020. "Nonmutual torques and the unimportance of motility for long-range order in two-dimensional flocks." *Physical Review E* 101(5): 052601. <https://doi.org/10.1103/PhysRevE.101.052601>

[^bialek2012]: Bialek et al. 2012. "Statistical mechanics for natural flocks of birds." *Proceedings of the National Academy of Sciences* 109(13): 4786–4791. <https://doi.org/10.1073/pnas.1118633109>

[^mora2016]: Mora et al. 2016. "Local equilibrium in bird flocks." *Nature Physics* 12(12): 1153–1157. <https://doi.org/10.1038/nphys3846>

[^ling2019b]: Ling et al. 2019. "Behavioural plasticity and the transition to order in jackdaw flocks." *Nature Communications* 10(1): 5174. <https://doi.org/10.1038/s41467-019-13281-4>

[^evangelista2017]: Evangelista et al. 2017. "Three-dimensional trajectories and network analyses of group behaviour within chimney swift flocks during approaches to the roost." *Proceedings of the Royal Society B: Biological Sciences* 284(1849): 20162602. <https://doi.org/10.1098/rspb.2016.2602>

[^ling2019c]: Ling et al. 2019. "Local interactions and their group-level consequences in flocking jackdaws." *Proceedings of the Royal Society B: Biological Sciences* 286(1906): 20190865. <https://doi.org/10.1098/rspb.2019.0865>

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
