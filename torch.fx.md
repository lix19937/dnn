# torch fx

## 本文要点：
* FX是什么？
* FX有什么作用？

https://zhuanlan.zhihu.com/p/587321150   

FX 是一个供开发者变换nn.Module实例的工具，主要包含3个组件：符号跟踪，中间表达，Python代码生成。这些组件的实际操作的一个演示：

* 符号跟踪：以符号的方式执行 Python 代码。通过给 nn Module 提供伪值，并记录在伪值上涉及到的操作。
* 中间表达：符号跟踪期间记录操作的容器，包含一系列节点（Node），节点包含了输入（placeholder），调用点（call_function、call_module、call_method），返回值。Graph是FX中间表达中使用的主要数据结构，Graph包含Node。
Graph is a data structure that represents a method on a GraphModule. 我们从Graph中可以得到3个信息：
1，method的输入：method的输入被指定为placeholder nodes
2，method里运行了哪些操作：get_attr, call_function, call_module, call_method 
3，method的返回值（输出）：返回值被指定为输出
Python代码生成：进行 Python到Python或Module到Module变换的工具。对于每一个中间表达，可以构建符合图语义的、有效的Python代码。此功能包含在GraphModule中，GraphModule是一个fx.Graph.生成的torch.nn.Module实例，包含了Graph以及从Graph生成的前向方法。
FX的Python到Python变换流程有以下4个模块组成：符号跟踪，中间表达，变换，Python代码生成。同时，这些模块可以独立使用，例如符号跟踪可以被单独使用，捕获某种形式的代码进行分析，代码生成可以被用来以编程方式生成模型，例如从配置文件生成模型。

## 变换：常见形式的函数如下 
 
上述函数中torch.nn.Module作为输入，获得fx.Graph，并进行修改，最终返回一个新的torch.nn.Module。

直接图操作
替换 function   

子图重写replace_pattern()   

Proxy     
图操作的另一种方式是 Proxy 机制进行 retracing，它可以自动化图rewriting，避免显示图修改，可以用 Python 函数的形式描述图重写规则。这里的关键是，把节点包装进 Proxy，传给图变换函数，返回新的 Proxy，用新的 Proxy 中的节点继续组建新图。要使用此方法，我们将要插入的操作编写为常规PyTorch代码，并使用Proxy对象作为参数调用该代码。这些Proxy对象将捕获对它们执行的操作，并将它们附加到图中。

## 调试   
避免图变换出错和一些调试建议：   
不要使用 Python 中的set() 管理 Node，集合类型是无序的；  
使用torch.allclose() 对比变换前后 Module 的结果；   
使用import pdb; pdb.set_trace() 在变换后的 Module 执行前暂停，然后单步调试；   
继承原 Module，把生成的forward() 函数复制粘贴到继承的 Module 中，用继承的 Module 调试；   
使用GraphModule.to_folder() 将 FX 代码导出到本地，然后导入模块进行调试；   
检查.graph 和 .code 属性，以及graph.print_tabular()；   

## 动态控制流    
符号跟踪的主要限制是它目前不支持动态控制流。也就是说，循环或if语句的条件可能取决于程序的输入值。   
if语句的条件依赖于x.sum()的值，该值依赖于函数输入x的值。由于x可以改变（如果新的输入张量传递给被跟踪函数），这就是动态控制流。  

## 静态控制流   
静态控制流是循环或if语句，其值不能在调用之间更改。通常，在PyTorch程序中，这种控制流是为了基于超参数对模型的结构做出决策而产生的。   
if self.do_activation不依赖于任何函数输入，因此它是静态的。do_activation可以被认为是一个超参数，并且对于该参数具有不同值的MyModule的不同实例的跟踪具有不同的代码。这是一个符号跟踪支持的有效模式。  
动态控制流的许多实例在语义上是静态控制流。这些实例可以通过删除对输入值的数据依赖性来支持符号跟踪，例如，通过将值移动到Module属性或在符号跟踪期间将具体值绑定到参数。  
在真正的动态控制流的情况下，包含此代码的程序部分可以作为对方法（请参阅使用Tracer类自定义跟踪）或函数（请参阅wrap()）的调用进行跟踪，而不是通过它们进行跟踪。   

## 非torch函数   
FX使用__torch_function__作为拦截调用的机制。有些函数，如内置Python函数或数学模块中的函数，不在__torch_function__中，但我们仍然希望在符号跟踪中捕获它们。  

## 符号跟踪的限制   
FX使用符号跟踪系统（也称为符号执行）以可转换/可分析的形式捕获程序的语义。该系统是跟踪程序（实际上是torch.nn.Module或函数）运行，并来记录操作。在执行过程中流经程序的数据不是真实的数据，而是符号。
不支持动态控制流（dynamic control flow），支持静态控制流，即 if/loop 的条件不随输入 tensor变化，生成的静态控制流代码，图是特化和展开的，不包含控制语句。语义上是静态控制的动态控制流，可以通过 symbolic_trace() 的concrete_args 参数传入具体参数。真实的动态控制流可以通过wrap() 避免 trace 它们；
torch.fx 目前支持torch, operator, math 3个 Python 包下面的函数，除此之外的函数需要用torch.fx.wrap() 将其声明为直接调用，例如 Python 内建函数len()；
创建 Tensor 的 API 不可被 trace，例如torch.ones(), torch.randn()，后者是非确定性的，需要wrap()；
符号跟踪过程中捕获的flag 变量，不可在执行阶段改变。例如，捕获 training 时的 dropout 函数torch.nn.functional.dropout(x, training=self.training)，以 eval 模式执行会出错，应该使用torch.nn.Dropout，它是叶子模块，内部实现不会被符号跟踪；

## 自定义Tracer类  
Tracer类是符号跟踪实现的基础类。可以通过子类Tracer来定制跟踪的行为，如   

## 叶子模块
叶模块是指在符号跟踪中显示为调用而不是进入跟踪的模块。叶模块的默认集合是标准torch.nn模块实例的集合。   

## 其它限制   

## 与torch   

## 几个重要的API 或类    
```
torch.fx.symbolic_trace  
torch.fx.Tracer
Tracer is the class that implements the symbolic tracing functionality of torch.fx.symbolic_trace. A call to symbolic_trace(m) is equivalent to Tracer().trace(m).
Tracer can be subclassed to override various behaviors of the tracing process. The different behaviors that can be overridden are described in the docstrings of the methods on this class.

torch.fx.Graph

torch.fx.wrap
```

## REF
https://arxiv.org/abs/2112.08429  
https://pytorch.org/docs/stable/fx.html   
https://pytorch.org/blog/optimizing-production-pytorch-performance-with-graph-transformations/   
