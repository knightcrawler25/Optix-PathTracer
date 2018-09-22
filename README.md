
OptixPathTracer
==========

![Dining Room](./dining.png)
[Dining Room](https://www.blendswap.com/blends/view/86457)

A physically based path tracer with support for Disney BRDF.

This was created by putting together bits and pieces from the Nvidia's Optix Advanced Samples Introduction tutorials.
The feature set is very basic since this is just a learning excercise, so if you see horribly written code or things that
can be done in a much better way please do share :)

Features
--------
( Almost all were part of existing code from the Nvidia's Optix Advanced Samples Repository on Github )  
- Unidirectional Path Tracing  
- Disney BRDF  
- Simple Glass BTDF  
- Sphere and Rect lights  
- Multiple Importance Sampling  
- Mesh Loading
- Simple Scene File (Basically stolen from Miles Macklin's excellent [Tinsel renderer](https://github.com/mmacklin/tinsel) ) so all credits go to him.

For the modified .obj files that go with the scene files, unzip the data.rar file in src/data/ into the same folder

Following are some scenes rendered with the path tracer
--------
[Bedroom](https://benedikt-bitterli.me/resources)
![Bedroom](./bedroom.png)

[Spaceship](https://www.blendswap.com/blends/view/75686) (Render Time: ~6 minutes on a GTX 750ti)
![Spaceship](./spaceship.png)

[Stormtrooper](https://www.blendswap.com/blends/view/76637) (Render Time: Render Time: ~4 minutes)
![Stormtrooper](./stormtrooper.png)

[Coffee Pot](https://benedikt-bitterli.me/resources) (Render Time: 4 minutes. 1k spp)
![Coffee Pot](./coffee_pot.png)

Remake of the Disney Hyperion Scene (Render Time: Quite long)
![Hyperion Scene](./hyperion.png)

![Dragon closeup](./dragon.png)

Thanks to Simon Kallweit for helping me out with the importance sampling code. He also has a nice write up of his [implementation](http://simon-kallweit.me/rendercompo2015/)

Models are from [Benedikt Bitterli's Rendering Resources](https://benedikt-bitterli.me/resources).

