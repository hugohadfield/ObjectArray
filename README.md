# ObjectArray
This is a pure python, dependency free, generic object array based on nested lists

It is designed for use with manipulating dense nested arrays of objects that define their own multiplication addition rules etc that may or may not be commutative.

It is slow and has very few checks for data input consistency.

Currently implemented are n-dimensional left/right sided correlation and convolution methods, an elementwise map method and a couple of other utility methods.

Next step is to implement numpy like slicing.
