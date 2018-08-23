# Fog
Fog is a java-baesd framework for deep learning (Java-1.8). It was done by Feng Sun and Yuyan Wang.

## Some questions about Fog.
### 1.Why use Java programming frameworks?
I had been practicing Java for two years, I am familiar with Java. Programming language is just a tool, don't care what use language coding algorithm. But to be honest, Python is much better at matrix support than Java. In the actual production process, we usually use Python, Lua and Matlab for deep learning programming.
### 2.Why do many loops in the code?
We created 'Fog' without considering the inefficiencies of Java (Not in the broad sense). Because java would not be the mainstream of deep learning in the future. We want to use this framework to show some the classical algorithms in deep learning. In addition, Java programs run on the Java virtual machine (JVM) and are limited by the resources acquired by the JVM. If we carefully adjust the code, we also need to adjust the virtual machine parameters of the framework runtime. It makes us feel like we're putting the cart before the horse.
