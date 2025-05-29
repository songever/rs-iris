mod logistic_reg;
mod bayes;
mod kmeans;
mod svm;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // create dummy classes 0 and 1
    svm::run()
}
