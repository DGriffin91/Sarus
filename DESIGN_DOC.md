Example of subgraph & nodes with metadata. Metadata starts with @ and is in [toml](https://docs.rs/toml/0.5.8/toml/) format.

```c#
@ lowshelf node 
# Uses toml for metadata config
description = "generate the coefficients for a 2nd order IIR shelf filter"
 
[inputs]
# each input can have settings for the ui knobs. Each property is optional.
# similar to in blender, users should be able to manually enter values outside of min/max
cutoff_hz = {default = 1000.0, min = 0.0, max = 20000.0, unit = "Hz", gradient = "TBD", label = "Hz", description = "the -3dB point"}
gain_db = {default = 0.0, min = -48.0, max = 48.0, unit = "dB", gradient = "TBD", label = "gain", description = "gain in dB of the shelf"}
q_value = {default = 1.0, min = 0.01, max = 100.0, gradient = "TBD", label = "Q", description = "filter sharpness"}
@
fn lowshelf(cutoff_hz, gain_db, q_value) -> (a1, a2, a3, m0, m1, m2) {
    cutoff_hz = min(cutoff_hz, globals.sample_rate_hz * 0.5)
    a = pow(10.0, gain_db / 40.0)
    g = tan(PI * cutoff_hz / globals.sample_rate_hz) / sqrt(a)
    k = 1.0 / q_value
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    m0 = 1.0
    m1 = k * (a - 1.0)
    m2 = a * a - 1.0
}
 
// Nodes can access persistent data
filter {
    ic1eq
    ic2eq
}
 
// There can also be anonymous nodes with no metadata 
fn filter(self, audio, a1, a2, a3, m0, m1, m2) -> (audio) {
    v3 = audio - self.ic2eq
    v1 = a1 * self.ic1eq + a2 * v3
    v2 = self.ic2eq + a2 * self.ic1eq + a3 * v3
    self.ic1eq = 2.0 * v1 - self.ic1eq
    self.ic2eq = 2.0 * v2 - self.ic2eq
    audio = m0 * audio + m1 * v1 + m2 * v2
}
 
//This sub graph would be audio generated based on the graph in the UI
@ lowshelf subgraph 
description = "a subgraph that implements a stereo shelf"
[inputs]
# derive the descriptions from the other node
cutoff_hz = {derive = "lowshelf"}
gain_db = {derive = "lowshelf"}
q_value = {derive = "lowshelf"}
[nodes]
lowshelf0 = {x = 10, y = 10}
filter0 = {x = 40, y = 10}
filter1 = {x = 40, y = 40}
@
eq_shelf {
    filter0
    filter1
}
fn stereo_shelf(self, audio_l, audio_r, cutoff_hz, gain_db, q_value) -> (audio_l, audio_r) {
    vlowshelf0_a1, vlowshelf0_a2, vlowshelf0_a3, vlowshelf0_m0, vlowshelf0_m1, vlowshelf0_m2 = lowshelf(cutoff_hz, gain_db, q_value)@lowshelf0
    audio_l = filter(self.filter0, audio_l, vlowshelf0_a1, vlowshelf0_a2, vlowshelf0_a3, vlowshelf0_m0, vlowshelf0_m1, vlowshelf0_m2)@filter0
    audio_r = filter(self.filter1, audio_r, vlowshelf0_a1, vlowshelf0_a2, vlowshelf0_a3, vlowshelf0_m0, vlowshelf0_m1, vlowshelf0_m2)@filter1
}
```