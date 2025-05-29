JNU人工智能课程论文

data目录存放了Iris数据集
src目录中实现了bayes、kmeans、logistic_reg、svm四个mod，分别对应四种统计学习方法
iris_data是用于数据处理的一个crate

在src的main.rs中的main函数中修改可运行4种算法之一，在配置了Rust运行环境的终端输入cargo run即可编译运行
示例如下
```rust
  fn main() -> Result<(), Box<dyn std::error::Error>> {
    svm::run()
}
```
