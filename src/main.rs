mod sample_cli;

#[tokio::main]
async fn main() {
    if let Err(error) = sample_cli::run().await {
        eprintln!("error: {}", error);
        std::process::exit(1);
    }
}
