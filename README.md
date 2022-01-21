# Incompressible Schrödinger FLow in real-time in TouchDesigner
original paper by Albert Chern et al.: https://cseweb.ucsd.edu/~alchern/projects/SchrodingersSmoke/

CUDA enabled GPU required. Built in TouchDesigner version 2021.15800


## Usage
Open a .toe file from the ISF/TouchDesignerProjects folder on a windows machine with CUDA enabled GPU.  
Press shift+r to reset the simulation.  
Press F1 to go into *perform mode*  
Press esc to leave *perform mode*  
In *perform mode* you can use your mouse to pan the camera.  
  
The interactive_prescribed_velocity.toe expects you to go into *perform mode*. The simulation will then react to mouse movement.  
  
Most relevant parameters can be found in the (red colored) *ISF* component under the *General* tab.  

##
The particles are sorted using a technique by Vincent Houzé: https://github.com/vinz9/CudaSortTOP

##
Please don't publish any results without my permission.

Feedback to contact.pelz@gmail.com, instagram.com/josefluispelz or twitter.com/josefluispelz
