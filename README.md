# Fog
Fog is a java-baesd framework for deep learning (Java-1.8). It was done by Feng Sun and Yuyan Wang.

## Some questions about Fog.
### 1.Why use Java programming frameworks?
I had been practicing Java for two years, I am familiar with Java. Programming language is just a tool, don't care what use language coding algorithm. But to be honest, Python is much better at matrix support than Java. In the actual production process, we usually use Python, Lua and Matlab for deep learning programming.
### 2.Where can I review API about 'Fog'? Have Chinese version?
You can see it in README.md. We provide the Chinese version, the second part in the article.
### 3.How to use the 'Fog'
You only need to download the source of code or download the 'Fog.jar' we provided.
### 4.Why do many loops in the code?
We created 'Fog' without considering the inefficiencies of Java (Not in the broad sense). Because java would not be the mainstream of deep learning in the future. We want to use this framework to show some the classical algorithms in deep learning. In addition, Java programs run on the Java virtual machine (JVM) and are limited by the resources acquired by the JVM. If we carefully adjust the code, we also need to adjust the virtual machine parameters of the framework runtime. It makes us feel like we're putting the cart before the horse.
### 5.Will you continue to maintain this program?
No.
## API
### Matrix (math.Matrix)
#### math.Matrix(int height, int width, String...type)
parameters:<br/>
height: The height of matrix.<br/>
width: The width of matrix.<br/>
type: Type is an optional parameter. You can choose 'random' or 'randn' or 'like someNumber'. 'random' means that the values in the matrix present a gaussian distribution. 'randn' means that the values in the matrix are randomly distribution in [0,1]<br/>

# Fog（中文版）
Fog是一个基于Java的深度学习傻瓜化框架，Java版本为1.8
## 一些关于Fog的问题
### 1.为什么使用Java编写此框架？
因为在之前的两年中，我通常使用Java作为主要的编程语言。语言只是一个工具，算法的实现是不在乎语言的。但是在矩阵的支持上，肯定是Matlab>Python>Java（虽然Matlab是工具而另外两者是语言，但也在此做一下比较）。对于Java来说，几乎没有关于矩阵的支持。所以我们只是使用自己熟悉的语言进行编写。没有任何语言好坏的抉择。
### 2.在哪里可以看到Fog的操作文档？
您可以在README.md中可以看到文档。
### 3.如何使用Fog？
您可以下载源码到自己的项目中，也可以下载我们提供的Fog.jar，让您的项目引用jar即可。
### 4.为什么在源码中出现很多循环体？
因为Java不提供任何关于矩阵的操作，所以我们自行实现了大部分矩阵的常用操作。在矩阵的实现中，常常需要遍历元素，所以会出现大量循环。但是也可以不用循环实现，但是我们并没有在Fog中实现这样的操作，因为我们的初衷只是借用Java实现经典的算法，而没有想要解决效率问题，在实际工程中，我们也不会使用Java作为开发主要语言，过去长期内没有，未来短期内也不会。并且Java运行于虚拟机之上，资源受限于虚拟机，如果细细调节效率问题，还需要调节Fog在运行时的虚拟机参数，在这上面花费大量时间对我们而言是不值得的。
### 5.未来还会维护Fog框架吗？
不会。因为Fog只是一个兴起之作，我们没有调用任何第三方库文件，全部由基础方法实现。对我们来说是一个挑战，我们乐于挑战这个难题，但是解决这个难题之后，维护会是一个需要耗费大量时间的事情，有些得不偿失。
