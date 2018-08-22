# Fog
&nbsp;&nbsp;&nbsp;&nbsp;Fog is a java-baesd framework for deep learning (Java-1.8). It was done by Feng Sun and Yuyan Wang.

# Some question about Fog.
&nbsp;&nbsp;&nbsp;&nbsp;<h4>Why do many loops in the code?</h4><br/>
&nbsp;&nbsp;&nbsp;&nbsp;We created 'Fog' without considering the inefficiencies of Java (Not in the broad sense). Because java would not be the mainstream of deep learning in the future. We want to use this framework to show some the classical algorithms in deep learning. In addition, Java programs run on the Java virtual machine (JVM) and are limited by the resources acquired by the JVM. If we carefully adjust the code, we also need to adjust the virtual machine parameters of the framework runtime. It makes us feel like we're putting the cart before the horse.
