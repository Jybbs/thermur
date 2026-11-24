/**
 * A flight the Lookout plays: its drones' positions and velocities, each
 * drone's east, north, and height at three places of each in meters, and the
 * time it has flown, moved on by `play`.
 */
export interface Playing {
  readonly count      : number
  readonly positions  : ArrayLike<number>
  readonly seconds    : number
  readonly velocities : ArrayLike<number>
  play : (dt: number) => void
}

/**
 * The frames of one flight the package flew through its graph, as `thermur
 * watch` serves them: the flock's size, the seconds between two frames, and
 * each frame's positions and velocities in turn, laid out as one flock's are.
 */
export interface Watched {
  drones     : number
  frame_s    : number
  positions  : readonly number[]
  velocities : readonly number[]
}
