# VI_WCK.jl
Julia codes for the examples in

__A Projection Anderson Acceleration Algorithm for Non-quasi-monotone Variational Inequalities with Hölder Continuous Terms__

by Chen, Kelley, and Wang

__You must run the notebook to view the examples.__

 ## Prerequisites
You must know enough Julia to 
- install packages and use Pkg,
- run an IJulia notebook,
- and manage Github.

## Dependencies ##

__Use the latest version of Julia (1.13) with these codes.__

You'll need to have these packages and their dependencies.
- LaTeXStrings
- Printf
- PythonPlot
- SIAMFANLEquations

## Running the notebook ##
- Clone this repository
- Put the directory in your Julia LOAD_PATH
- cd to the directory; start Julia, and then type
   ```notebook(;dir=pwd())``` in the REPL
- a web page of the directory will open. Click on the notebook __Examples.ipynb__
- When the notebook opens, do a ```run all```
- If you get error messages about the kernel:
  - Shut down Julia and the notebook
  - Restart Julia and do ```build IJulia``` from the package manager
  - start the notebook
  
