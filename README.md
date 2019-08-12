# Gaia3Dmapping
Summer project at ITA summer 2019
The goal of the project is to make 3D maps of the stellar data form the Gaia DR2. This is done using Healpix coordinates and visualized in Mollveide projection maps, where each map contain the stars within some distance span, e.i. 1000 pc to 2000 pc.

There are three main programs, one for reading and writing the parameter files form the data files in the Gaia DR2.
The next program is for converting some of the parameter files to astrophysical parameters, like Healpix coordinates and distance from the parallax. The third program are the map making program, it has functions to create maps containing all stars, make histogram over the stellar distances, check the effect of different Nside. And make the map layers to form a 3D projection of the stars. Also are there two helping programs, one with different usefull global parameters and one program with different calculation tools.

#### Programs contained:
 - Datahandling.py
 - AstroData.py
 - DistanceMapping.py
 
 - Tools.py
 - params.py

#### link to Gaia documentation:
https://gea.esac.esa.int/archive/documentation/index.html
