
from generator import generate_synthetic, save_to_bin

if __name__ == "__main__":
    print("Generating 24 hours (1440 minutes) of synthetic telemetry...")
    # Inject anomalies so the model (and verification) has something interesting
    df = generate_synthetic(n_minutes=1440, inject_anoms=True)
    
    output_path = "data/telemetry.bin"
    save_to_bin(df, output_path)
    print(f"Saved synthetic telemetry to {output_path}")
