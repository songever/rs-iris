mod logistic_reg;
mod bayes;
mod kmeans;
mod svm;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    svm::run()
}
