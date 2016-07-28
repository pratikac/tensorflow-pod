tensorflow-pod
==============


This is a [pods](https://sourceforge.net/projects/pods/) wrapper for tensorflow. The aim
is to be able to plug-in tensorflow easily into the build system of other projects. Instead
of building tensorflow (which is an involved process), we will simply install the headers and
copy over a compiled library.

``patch.diff`` in this directory is only needed to build tensorflow for the GPU.
