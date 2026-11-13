# The Interaction Rules

Each drone in *Thermur* steers by the sum of a few rules, and each rule reads only what that one drone senses. The controller has a slot for each part of the motion, meaning the heading a drone turns toward, the rate it aligns with each neighbor at, the random increments driving it, the force holding it at a spacing, the contact keeping two drones apart, and the roost holding the flock over one place. Each slot holds a rule borrowed from another physical system, a starling flock, a jamming solid, a school of fish, or a swarm of active particles. An open slot starts from a default and flies each candidate rule against it, keeping a candidate only where the measurements favor it. In the flights below the elastic springs and the fitted attraction each pass that test, and every slot still holds its default.

## The Neighbors a Drone Senses

A drone senses its seven nearest neighbors by rank rather than within a fixed range, the topological interaction measured in starling flocks[^ballerini2008a]. For each neighbor it reads the offset, the distance, and the velocity as its radio last heard them, together with the velocity a neighbor held half a second before that for the rule that reads a delay, and it reads its own position, its own command, and the excess temperature and the wind of the air at its position. A rule receives that view and nothing else.

By rank a drone always senses seven neighbors, whereas the number within a fixed range swings as a flock gathers and splits. The drawing below flies the front page's illustration flock rather than the controller, the pointer its falcon, so it shows how the two counts differ and nothing about how the controller's drones fly.

<!-- neighbors -->

The suite checks this with a property test, which moves a drone that another drone does not sense to a random place still outside that drone's neighbors and asserts that the second drone's acceleration and the amplitude of its noise stay the same, under the default composite and under every candidate.

## The Flight

A flight integrates each drone's command under the sum of the slots, and its velocity and position under the plant and the wind below, through the Euler-Maruyama scheme, one step of diffrax's Euler solver at a time, its noise the Brownian increment a virtual Brownian tree draws, which carries the $`\sqrt{\Delta t}`$ scaling of a stochastic differential equation[^higham2001]. Every draw derives from one seed, and several seeds fly together as one batch. The flock launches in a thin horizontal cloud, its axes in the proportion 5.6 to 2.8 to 1 measured on starling flocks, its thin axis vertical[^ballerini2008b]. Each drone commands a constant cruise speed, so under a heading rule that holds the speed, the part of the acceleration along a drone's command is removed and the command returns to the cruise speed after every step.

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
| `spacing_m` | `15.0` | above 0 | The launch cloud's cube root of the volume per drone, in meters |
| `step_s` | `0.02` | above 0 | The step of the integrator, in seconds |
| `turn_at_s` | `25.0` | at least 0 | The moment the first drone starts the turn, in seconds |
| `turn_deg` | `0.0` |  | The angle the first drone turns through and then holds, in degrees counterclockwise, none by default |
| `turning_deg_s` | `2.0` | above 0 | The rotation of the flock's mean heading, over `turning_window_s`, past which a spontaneous turn counts as started, in degrees per second |
| `turning_window_s` | `5.0` | above 0 | The time over which the flock's mean heading or a drone's heading is read for a turn, in seconds, longer than a step's noise on either |

<!-- /plan -->

## The Plant and the Wind

A drone's autopilot carries its velocity through the air toward its command over a first-order lag of one second[^viragh2014], holding the command within the speeds and the response within the accelerations the PX4 multicopter defaults set, with the horizontal speed raised within PX4's range to the 15.2 m/s of the fastest starling flock measured[^ballerini2008b]. The plant holds no gravity term, since the autopilot carries the drone's weight. A drone moves over the ground at its velocity through the air plus the wind at its position[^berman2021], so the WRF-SFIRE winds carry the flock in place of a drag force, whereas the parametric threat's air is still. The table is written from the plant's record, set under `[tool.thermur.plant]` or through a `--plant.` flag, and `mise render` rewrites it.

<!-- plant -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `climb_m_s` | `3.0` | above 0 | The fastest climb a command holds, in meters per second, PX4's `MPC_Z_VEL_MAX_UP` |
| `climbing_m_s2` | `4.0` | above 0 | The largest upward acceleration, in meters per second squared, PX4's `MPC_ACC_UP_MAX` |
| `descending_m_s2` | `3.0` | above 0 | The largest downward acceleration, in meters per second squared, PX4's `MPC_ACC_DOWN_MAX` |
| `descent_m_s` | `1.5` | above 0 | The fastest descent a command holds, in meters per second, PX4's `MPC_Z_VEL_MAX_DN` |
| `lag_s` | `1.0` | above 0 | The time constant over which a drone's velocity relaxes toward its command, in seconds |
| `level_m_s` | `15.2` | above 0 | The fastest horizontal speed a command holds, in meters per second, within the 0 to 20 PX4 allows its `MPC_XY_VEL_MAX` |
| `level_m_s2` | `5.0` | above 0 | The largest horizontal acceleration, in meters per second squared, the ceiling PX4's `MPC_ACC_HOR_MAX` sets in its direct-velocity mode |

<!-- /plant -->

## The Radio

A drone hears its neighbors only through its radio, which broadcasts each drone's state five times a second and delivers it one second late, the sensor update period and the delay of the setup the outdoor drone flock was simulated under[^vasarhelyi2018], whose update period matches the earlier quadcopter flock's[^viragh2014]. A packet arrives only where its sender sat within 80 m of the drone as it was sent, the communication range of that setup. A drone ranks its seven nearest neighbors by the positions it heard, as a starling interacts with a fixed number of the neighbors it perceives[^ballerini2008a], so a neighbor heard from within range may since have flown past it, and a drone that has heard fewer than seven others reads each missing neighbor as one sitting on the drone at its own command. Each packet from each sender is lost with a chance of 0.4, the middle of the 20 to 60% the outdoor flock's radios lost, and a drone keeps the last state it heard from a sender until another packet arrives. The replay that scores each composite's drift delivers every packet sent from within range, keeping the delay. The table is written from the radio's record, and `mise render` rewrites it.

<!-- radio -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `delay_s` | `1.0` | at least 0 | The age of a broadcast as it arrives, in seconds |
| `loss` | `0.4` | at least 0 and below 1 | The share of the packets between two drones the link loses, each loss drawn apart from every other |
| `range_m` | `80.0` | above 0 | The distance between two drones, as a packet is sent, past which the packet never arrives, in meters |
| `rate_hz` | `5.0` | above 0 | The number of packets each drone sends each second |

<!-- /radio -->

## The Roost

The roost slot holds the flock over a cylinder around a roost through the roost term of a starling flock model[^hildenbrandt2010] at drone scale. Outside the cylinder its horizontal part turns a drone toward the side nearer the roost, at the full turning acceleration where the drone heads straight out and none where it heads straight in, and its vertical part is a spring toward the roost's height at that model's vertical weight over its bird's mass, 0.2 N per meter over 80 g, which the model calibrates to keep the flock from flying off vertically. The spring is critically damped, at twice the square root of its stiffness, since an undamped spring behind the plant's lag grows unstable. The roost's center, radius, and height are fields of the slot, so a flight sets where the roost stands. The table is written from the slot's record, and `mise render` rewrites it.

<!-- roost -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `center_m` | `[600.0, 1000.0]` |  | The roost's east and north, in meters, by default on the parametric threat's course from the launch to the domain's western edge |
| `height_m` | `100.0` |  | The height the drone is pulled toward, in meters above the ground |
| `radius_m` | `600.0` | above 0 | The radius of the cylinder, in meters |
| `stiffness_per_s2` | `2.5` | at least 0 | The acceleration toward the height per meter away from it, in reciprocal square seconds, by default the starling model's vertical weight of 0.2 newtons per meter over its bird's mass of 80 grams |
| `turning_m_s2` | `5.0` | at least 0 | The acceleration turning a drone headed straight out of the cylinder, in meters per second squared, half of it for a drone headed along the cylinder's edge, by default the ceiling PX4's `MPC_ACC_HOR_MAX` sets in its direct-velocity mode |

<!-- /roost -->

Under the defaults, 100 drones over four seeds for 120 s keep their nearest neighbors 22.7 m away, inside the 12 to 30 m the outdoor flock kept[^vasarhelyi2018], at a polarization of 0.971. The vertical spring holds the drones' heights to a standard deviation of 2.0 m about the roost's, so the flock's axes stand at 1 to 50 to 77, a sheet far flatter than the 1 to 2.8 to 5.6 of starling flocks[^ballerini2008b]. Over a 300 s flight the flock's center runs about 300 to 510 m past the cylinder before it turns back, and three of the four seeds end the flight within 120 m of its edge.

## Contact

Two drones closer than one diameter push apart through a harmonic soft-sphere contact, whose force grows linearly with the overlap and stays bounded however far two drones overlap[^ohern2003]. The contact is fixed rather than open. The table is written from the rule's record, and `mise render` rewrites it.

<!-- contact -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `diameter_m` | `6.0` | above 0 | The distance within which two drones push apart, in meters |
| `stiffness_m_s2` | `8.0` | at least 0 | The acceleration at full overlap, in meters per second squared |

<!-- /contact -->

## Heading

The default heading rule is the overdamped limit of the inertial spin model at a constant speed[^cavagna2015], in which a drone's velocity relaxes toward each neighbor's at the rate the coupling sets, with no spin carried between steps. The candidates are a spinless second-order consensus, which leaves the speed free and relaxes a drone's velocity toward its neighbors' mean as well, where the source relaxes toward the whole flock's[^lizarraga2026], and alignment on the neighbors' velocities half a second old[^geiss2022]. The trial turns one drone through 20° and through 90° and measures how far the turn reaches across the flock and how fast it crosses, beside the polarization and the frequency at which the spontaneous velocity correlation $`C(k,\omega)`$ peaks. The table is written from each rule's record, the default declaring no parameter, and `mise render` rewrites it.

<!-- heading -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `consensus.centering_per_s` | `0.1` | at least 0 | The rate at which a drone's velocity relaxes toward its neighbors' mean, in reciprocal seconds |
| `delayed.lag_s` | `0.5` | above 0 | The age of the neighbor states the slot reads, in seconds |

<!-- /heading -->

The chart below reads the share of the first drone's turn that the farthest of five shells of drones turned through under each heading rule, the mean across four seeds beside its standard error, from a run of `thermur fly coupling_trial heading_trial` that `mise measure` wrote out, and `mise measure` redraws it from a newer run.

<!-- reach -->

Under every heading rule the farthest drones follow about a third of a 20° turn and less than a tenth of a 90° one, the delayed rule reaching furthest at 0.40 and 0.09 and the consensus rule least at 0.31 and 0.08. The standard errors on the small turn run wide, since four seeds separate the rules poorly there.

## Coupling

The default coupling holds one constant rate $`J`$ over every one of a drone's seven neighbors[^bialek2012]. The candidates are a rule that drops each of a drone's seven neighbors past a fixed range once the air at the drone runs hot, after the interactions jackdaws show while mobbing a predator[^ling2019b], a rate that decays with distance even within a rank, after the interactions by physical distance rather than by rank that chimney swifts show[^evangelista2017], and a rate stronger for a neighbor flying abeam than for one ahead or behind, carrying over to alignment the anisotropy of the attraction measured between jackdaws, stronger across their flight than along it[^ling2019c]. The trial flies the metric rule with its onset below any excess, so from launch it drops each of a drone's seven neighbors past 25 m, and flies 100 drones launched at spacings of 10, 15, and 25 m and measures how the polarization changes with the density and how alignment between two drones falls with the neighbor's rank. The table is written from each rule's record, and `mise render` rewrites it.

<!-- coupling -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `constant.coupling_per_s` | `0.6` | at least 0 | The rate at which a drone aligns with each neighbor, in reciprocal seconds |
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

The constant and the anisotropic rules hold the polarization above 0.92 at every spacing, whereas the metric rule, dropping every neighbor past 25 m, holds under 0.42 and falls to 0.23 at the sparsest, and the distance-weighted rule holds under 0.80 and loses more past the middle spacing. Between two measured spacings the chart draws the straight line between them, which no flight measured, and three spacings over four seeds leave each rule's curve coarse.

## Noise

The default noise drives every drone's command with one amplitude, and the polarization it is measured against is the $`0.96 \pm 0.03`$ of starling flocks[^cavagna2010]. The candidates are a random force that decorrelates over a persistence time[^caprini2021], noise that grows with the disagreement among a drone's neighbors, the noise of a finite number of interacting animals[^jhawar2020], and noise added to the neighbors' summed heading before it is normalized, whose influence falls as the neighbors agree[^chate2008]. The trial flies 30, 100, and 200 drones and measures the polarization against the flock's size and the rate at which the flock turns of its own accord. The table is written from each rule's record, and `mise render` rewrites it.

<!-- noise -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `white.amplitude` | `4.2` | at least 0 | The amplitude, in meters per second per root second |
| `intrinsic.copying_per_s` | `1.0` | at least 0 | The rate at which a drone copies a neighbor, in reciprocal seconds |
| `intrinsic.spontaneous` | `3.7` | at least 0 | The amplitude where every neighbor agrees, in meters per second per root second |
| `persistent.persistence_s` | `1.0` | above 0 | The time over which the force decorrelates, in seconds |
| `persistent.strength_m_s2` | `3.8` | at least 0 | The root-mean-square force per axis, in meters per second squared |
| `vectorial.amplitude` | `4.1` | at least 0 | The amplitude where the neighbors fly as one, in meters per second per root second |
| `vectorial.floor` | `0.3` | above 0 | The smallest length the mean neighbor heading divides by, near the length of the mean of a neighbor count of random headings |

<!-- /noise -->

## Cohesion

The default cohesion is a generalized Morse potential, a short-range repulsion against a longer-range attraction, summed over the seven neighbors. Its parameters sit outside the region where a swarm summing that potential over every pair collapses as it grows[^dorsogna2006], which holds where the repulsion's strength times the cube of its length exceeds the attraction's, and a rule built inside that region is refused. The candidates are linear springs of nonzero rest length to each neighbor, from active elastic matter[^ferrante2013], a spring toward the neighbors' centroid, the cohesion a vision-based drone flock flies[^schilling2021], and an attraction that depends on a neighbor's distance and bearing, as fitted to pairs of fish[^calovi2018]. A control flies the default with the coupling off, since attraction alone can order a flock[^caprini2023]. The trial flies 30, 100, and 200 drones and measures the nearest-neighbor distance against the flock's size, the density at the flock's border over that at its center, the height of the first peak of the pair distribution $`g(r)`$, and the number of groups a small flock splits into. The table is written from each rule's record, and `mise render` rewrites it.

<!-- cohesion -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `morse.attraction_m` | `12.0` | above 0 | The decay length of the attraction, in meters |
| `morse.attraction_m2_s2` | `80.0` | at least 0 | The strength of the attraction, in square meters per square second |
| `morse.repulsion_m` | `2.4` | above 0 | The decay length of the repulsion, in meters |
| `morse.repulsion_m2_s2` | `10400.0` | at least 0 | The strength of the repulsion, in square meters per square second |
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
| Cohesion | Morse, the default | *2.25 groups at 30 drones and 6.75 at 100, border density 0.74 of the center's at 100, $`g(r)`$ peak 2.06 times its mean* |
| Cohesion | Centroid spring | *1.25 groups at 30 and 2.75 at 100, border density 0.23, peak 2.28* |
| Cohesion | Elastic springs | *1.25 groups at 30 and 2.75 at 100, border density 0.54, peak 1.78* |
| Cohesion | Fitted attraction | *2.00 groups at 30 and 3.25 at 100, border density 0.66, peak 2.91* |
| Cohesion | Coupling off | *24.25 groups at 30, polarization 0.08 to 0.16* |
| Coupling | Constant, the default | *Polarization 0.951, 0.974, and 0.977 from the densest launch to the sparsest* |
| Coupling | Anisotropic | *Polarization 0.929, 0.961, and 0.963* |
| Coupling | Metric | *Polarization 0.411, 0.384, and 0.227* |
| Coupling | Weighted by distance | *Polarization 0.786, 0.795, and 0.727* |
| Heading | Overdamped, the default | *The farthest drones turning through 0.34 of a 20° turn and 0.08 of a 90° one, polarization 0.974* |
| Heading | Consensus | *0.31 and 0.08, polarization 0.927* |
| Heading | Delayed | *0.40 and 0.09, polarization 0.975* |
| Noise | White, the default | *Polarization 0.993, 0.974, and 0.951 at 30, 100, and 200 drones, no spontaneous turn at 100* |
| Noise | Intrinsic | *Polarization 0.993, 0.975, and 0.951, no turn* |
| Noise | Persistent | *Polarization 0.992, 0.979, and 0.954, no turn* |
| Noise | Vectorial | *Polarization 0.993, 0.978, and 0.952, no turn* |

The default flock does not hold together, splitting into 6.75 groups at 100 drones with its drones 22.9 m from their nearest neighbors against a launch spacing of 15 m, and it is less dense at its border than at its center, where starling flocks are denser at the border[^ballerini2008b]. The centroid spring and the elastic springs each keep a small flock nearly whole and pull its drones to within 12 m of their nearest neighbors, but the centroid spring thins the border to 0.23 of the center's density and fails, whereas the elastic springs pass on keeping the small flock whole. The fitted attraction passes on raising the peak of $`g(r)`$ to 2.91. A constant coupling keeps the polarization nearly flat as the density changes, as a topological rule does, while the metric and weighted rules lose order as the flock thins. The speed of a turn reads zero in every flight but two, one default flight of the 20° turn at 9.8 m/s and one delayed flight of it at 3.9 m/s, since in every other flight no three of the five shells of drones around the turning drone turned through half of the turn at distinct times, so no fit through their times gives an outward speed. No heading rule reaches further than the default on either turn by more than the standard error, so none passes. No flight under any noise turns spontaneously, so the noises separate on the polarization alone, and none sits nearer the 0.96 of starling flocks than the default by more than the standard error. On every replayed flight but the uncoupled one, the composite that flew it ranks first by one-step log-likelihood or within 0.001 of first, and the uncoupled flight ranks the elastic springs first by 0.87. The margin over the default reaches 2.8 to 3.1 where a spring pulls toward the neighbors and 0.26 under the persistent force, both of which the drift carries, though on each flight a spring flew the three springs stay within 0.001 of one another. Between rules whose drifts differ less it stays under 0.11, since at one 0.02 s step per frame the Brownian increment dominates each step's residual. The noises that enter through the increment alone tie, because the replay reads only the drift.

[^ballerini2008a]: Ballerini et al. 2008. "Interaction ruling animal collective behavior depends on topological rather than metric distance: Evidence from a field study." *Proceedings of the National Academy of Sciences* 105(4): 1232–1237. <https://doi.org/10.1073/pnas.0711437105>

[^higham2001]: Higham 2001. "An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations." *SIAM Review* 43(3): 525–546. <https://doi.org/10.1137/S0036144500378302>

[^ballerini2008b]: Ballerini et al. 2008. "Empirical investigation of starling flocks: a benchmark study in collective animal behaviour." *Animal Behaviour* 76(1): 201–215. <https://doi.org/10.1016/j.anbehav.2008.02.004>

[^viragh2014]: Virágh et al. 2014. "Flocking algorithm for autonomous flying robots." *Bioinspiration & Biomimetics* 9(2): 025012. <https://doi.org/10.1088/1748-3182/9/2/025012>

[^berman2021]: Berman et al. 2021. "Transport barriers to self-propelled particles in fluid flows." *Physical Review Fluids* 6(1): L012501. <https://doi.org/10.1103/PhysRevFluids.6.L012501>

[^vasarhelyi2018]: Vásárhelyi et al. 2018. "Optimized flocking of autonomous drones in confined environments." *Science Robotics* 3(20): eaat3536. <https://doi.org/10.1126/scirobotics.aat3536>

[^hildenbrandt2010]: Hildenbrandt et al. 2010. "Self-organized aerial displays of thousands of starlings: a model." *Behavioral Ecology* 21(6): 1349–1359. <https://doi.org/10.1093/beheco/arq149>

[^ohern2003]: O'Hern et al. 2003. "Jamming at zero temperature and zero applied stress: The epitome of disorder." *Physical Review E* 68(1): 011306. <https://doi.org/10.1103/PhysRevE.68.011306>

[^cavagna2015]: Cavagna et al. 2015. "Flocking and Turning: a New Model for Self-organized Collective Motion." *Journal of Statistical Physics* 158(3): 601–627. <https://doi.org/10.1007/s10955-014-1119-3>

[^lizarraga2026]: Lizárraga and de Aguiar 2026. "Collective turns in spinless flocks." *Physical Review E* 114(3): 035401. <https://doi.org/10.1103/6976-lcw7>

[^geiss2022]: Geiß et al. 2022. "Signal propagation and linear response in the delay Vicsek model." *Physical Review E* 106(5): 054612. <https://doi.org/10.1103/PhysRevE.106.054612>

[^bialek2012]: Bialek et al. 2012. "Statistical mechanics for natural flocks of birds." *Proceedings of the National Academy of Sciences* 109(13): 4786–4791. <https://doi.org/10.1073/pnas.1118633109>

[^ling2019b]: Ling et al. 2019. "Behavioural plasticity and the transition to order in jackdaw flocks." *Nature Communications* 10(1): 5174. <https://doi.org/10.1038/s41467-019-13281-4>

[^evangelista2017]: Evangelista et al. 2017. "Three-dimensional trajectories and network analyses of group behaviour within chimney swift flocks during approaches to the roost." *Proceedings of the Royal Society B: Biological Sciences* 284(1849): 20162602. <https://doi.org/10.1098/rspb.2016.2602>

[^ling2019c]: Ling et al. 2019. "Local interactions and their group-level consequences in flocking jackdaws." *Proceedings of the Royal Society B: Biological Sciences* 286(1906): 20190865. <https://doi.org/10.1098/rspb.2019.0865>

[^cavagna2010]: Cavagna et al. 2010. "Scale-free correlations in starling flocks." *Proceedings of the National Academy of Sciences* 107(26): 11865–11870. <https://doi.org/10.1073/pnas.1005766107>

[^caprini2021]: Caprini and Marini Bettolo Marconi 2021. "Inertial self-propelled particles." *The Journal of Chemical Physics* 154(2): 024902. <https://doi.org/10.1063/5.0030940>

[^jhawar2020]: Jhawar et al. 2020. "Noise-induced schooling of fish." *Nature Physics* 16(4): 488–493. <https://doi.org/10.1038/s41567-020-0787-y>

[^chate2008]: Chaté et al. 2008. "Collective motion of self-propelled particles interacting without cohesion." *Physical Review E* 77(4): 046113. <https://doi.org/10.1103/PhysRevE.77.046113>

[^dorsogna2006]: D'Orsogna et al. 2006. "Self-Propelled Particles with Soft-Core Interactions: Patterns, Stability, and Collapse." *Physical Review Letters* 96(10): 104302. <https://doi.org/10.1103/PhysRevLett.96.104302>

[^ferrante2013]: Ferrante et al. 2013. "Elasticity-Based Mechanism for the Collective Motion of Self-Propelled Particles with Springlike Interactions: A Model System for Natural and Artificial Swarms." *Physical Review Letters* 111(26): 268302. <https://doi.org/10.1103/PhysRevLett.111.268302>

[^schilling2021]: Schilling et al. 2021. "Vision-Based Drone Flocking in Outdoor Environments." *IEEE Robotics and Automation Letters* 6(2): 2954–2961. <https://doi.org/10.1109/LRA.2021.3062298>

[^calovi2018]: Calovi et al. 2018. "Disentangling and modeling interactions in fish with burst-and-coast swimming reveal distinct alignment and attraction behaviors." *PLOS Computational Biology* 14(1): e1005933. <https://doi.org/10.1371/journal.pcbi.1005933>

[^caprini2023]: Caprini and Löwen 2023. "Flocking without Alignment Interactions in Attractive Active Brownian Particles." *Physical Review Letters* 130(14): 148202. <https://doi.org/10.1103/PhysRevLett.130.148202>

[^torney2018]: Torney et al. 2018. "Inferring the rules of social interaction in migrating caribou." *Philosophical Transactions of the Royal Society B: Biological Sciences* 373(1746): 20170385. <https://doi.org/10.1098/rstb.2017.0385>

[^ferretti2020]: Ferretti et al. 2020. "Building General Langevin Models from Discrete Datasets." *Physical Review X* 10(3): 031018. <https://doi.org/10.1103/PhysRevX.10.031018>

[^gerardos2025]: Gerardos and Ronceray 2025. "Principled Model Selection for Stochastic Dynamics." *Physical Review Letters* 135(16): 167401. <https://doi.org/10.1103/ltdt-hvh7>
