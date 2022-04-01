# polygon_dust

## What it does

The basic idea behind the code is to estimate dust opacities and in turn dust masses within user-defined polygons. 
The basic underlying idea is that the edge of the polygon defines an unobscured region within a galaxy, and thus 
the lower average intensity within the polygon can be interpreted as attenuation due to dust. The relative intensity 
can then be used to calculate an optical depth, and from there we can estimate a gas column density. With known distance 
this then provides a measure of the total dust mass. This dust mass presents a lower limit to the actual dust mass, since 
we assume a foreground dust screen (i.e. dust could be hidden deeper inside the galaxy), and that the optical depth is 
optically thin (once it becomes optically thick we can not tell HOW optically thick it is). 


## How to use it, command line parameters

To execute, run

    python3 polygon2dust --region=your_region.reg image1.fits image2.fits --output=my_results --distance=50

The **--region** option required a region file with the polygon edges in the ds9 format. Best way to get this is either to
generate polygons by hand, or define a contour, and then use the "contour -> copy to polygons" option within ds9.

**--distance** takes the distance to your source in Mpc (Mega-Parsecs). This is required to convert column densities
(atoms per square centimeter) into actual atom numbers and from there into a dust mass.

**--output** accepts a basename for the output catalog. By default, two output files will be generated, with one in CSV 
format and another one as VO-Table. Filenames for these files are generated from the specified option, with the 
respective file extension appended (Example: --output=galaxy will result in a galaxy.csv and a galaxy.vot output).

