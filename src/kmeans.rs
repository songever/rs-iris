use linfa::traits::Fit;
use linfa::traits::Predict;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa::dataset::Dataset;
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_npy::write_npy;
use linfa_nn::distance::{LInfDist, L1Dist, L2Dist};
use iris_data::IrisRecord;
use rand_xoshiro::Xoshiro256Plus;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<IrisRecord> = iris_data::load_iris_data("./data/iris.csv")?;

    let n_samples = data.len();
    let n_features = 4;

    let features: Array2<f64> = Array2::from_shape_vec(
        (n_samples, n_features),
        data.iter()
            .flat_map(|r| r.into_feature_vector())  // 返回 Vec<f64>
            .collect::<Vec<f64>>(),
    )?;

    let labels: Vec<String> = data.iter()
        .map(|r| r.into_label()) // 返回 String 或 &str
        .collect();

    //创建 linfa 的 Dataset
    let dataset = Dataset::new(features, labels.into())
        .map_targets(|species| match species.as_str() {
            "Iris-setosa" => 1usize,
            "Iris-versicolor" => 2usize,
            "Iris-virginica" => 0usize,
            _ => panic!("Unknown species"),
    });

    let mut centroids = Vec::new();
    for i in &[0usize, 1, 2] {
        if let Some((x, _)) = dataset.sample_iter()
            .find(|(_, y)| *y.into_scalar() == *i) {
                centroids.push(x);
        }
    } 

    let rng1 = Xoshiro256Plus::seed_from_u64(42);
    let rng2 = Xoshiro256Plus::seed_from_u64(42);
    let rng3 = Xoshiro256Plus::seed_from_u64(42);
    // Configure our training algorithm
    let n_clusters = centroids.len();
    let model_l1:KMeans<_, _> = KMeans::params_with(n_clusters, rng1, L1Dist)
        .max_n_iterations(50)
        .tolerance(1e-5)
        .fit(&dataset)
        .expect("KMeans fitted");
    let model_l2:KMeans<_, _> = KMeans::params_with(n_clusters, rng2, L2Dist)
        .max_n_iterations(50)
        .tolerance(1e-5)
        .fit(&dataset)
        .expect("KMeans fitted");
    let model_linf:KMeans<_, _> = KMeans::params_with(n_clusters, rng3, LInfDist)
        .max_n_iterations(50)
        .tolerance(1e-5)
        .fit(&dataset)
        .expect("KMeans fitted");
    println!("Dataset: {}", dataset.targets());
    // Assign each point to a cluster using the set of centroids found using `fit`
    
    let result_l1 = model_l1.predict(dataset.clone());
    let result_l2 = model_l2.predict(dataset.clone());
    let result_linf = model_linf.predict(dataset.clone());
    println!("l1 Cluster assignments: {}", result_l1.targets());
    println!("l2 Cluster assignments: {}", result_l2.targets());
    println!("linf Cluster assignments: {}", result_linf.targets());
    
    let DatasetBase {
        records, targets, ..
    } = result_l1;
    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("clustered_dataset_l1.npy", &records).expect("Failed to write .npy file");
    write_npy("clustered_memberships_l1.npy", &targets.map(|&x| x as u64))
        .expect("Failed to write .npy file");

    let DatasetBase {
        records, targets, ..
    } = result_l2;
    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("clustered_dataset_l2.npy", &records).expect("Failed to write .npy file");
    write_npy("clustered_memberships_l2.npy", &targets.map(|&x| x as u64))
        .expect("Failed to write .npy file");

    let DatasetBase {
        records, targets, ..
    } = result_linf;
    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("clustered_dataset_linf.npy", &records).expect("Failed to write .npy file");
    write_npy("clustered_memberships_linf.npy", &targets.map(|&x| x as u64))
        .expect("Failed to write .npy file");
    Ok(())
}