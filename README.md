# TIM: Temporal Interference Memory

### Quantum-Inspired Hyperdimensional Fuzzy Search Engine

Zero-shot fuzzy matching on raw symbolic sequences using classical wave interference.

**No embeddings • No training • GPU-accelerated • Real-time**

## Key Innovations

- **Sinusoidal Temporal Encoding** → High-dimensional wave vectors
- **Resonance Node Indexing** → 90–99% search space pruning
- **OpenCL GPU Kernel** → Parallel encoding of millions of items
- **Phase-Aware Fidelity Scoring** → |⟨ψ|φ⟩|² with resonance bonuses

## Applications

- **TIMShield** – Phishing domain & obfuscated malware detection
- **TIM-BLAST** – Coming soon (DNA/protein fuzzy search)

## Performance (Real Benchmarks)

| Dataset           | Items   | Build Time | Search Time | Pruning % |
|-------------------|---------|------------|-------------|-----------|
| Phishing domains  | 1,000+  | 180 ms     | 12 ms       | 98.7%     |

## Quick Start

```csharp
var tim = new TimEngine(domains, isMalware: false);
tim.Build();
var results = tim.Search("p@ypal.com");

foreach (var r in results)
{
    Console.WriteLine($"{r.FoundItem} - Fidelity: {r.Fidelity:F2}%");
}
```

## Project Structure

```
TIM-Vector-Search/
├── src/
│   ├── TimCore/           # Core encoding logic
│   ├── Kernels/           # OpenCL .cl kernel files
│   └── TimShield/         # Cybersecurity app
├── examples/              # Demo scripts
├── benchmarks/            # Performance tests
├── Data/                  # CSV datasets
└── docs/                  # Mathematical documentation
```

## Building

```bash
# Requires Cloo (OpenCL wrapper) and .NET Framework
dotnet build
```

## Running

```bash
dotnet run
# Select: Domain or Malware
# Enter query string (e.g., "p@ypal.com")
```

## How It Works

1. **Encoding**: Convert each string to a high-dimensional sinusoidal wave vector
2. **Indexing**: Extract "resonance nodes" (high-magnitude dimensions) for pruning
3. **Search**: Compute wave similarity (fidelity) between query and candidates
4. **Scoring**: Rank by |⟨ψ|φ⟩|² with phase and resonance bonuses

## Requirements

- .NET Framework / .NET Core 6+
- OpenCL-capable GPU (NVIDIA/AMD/Intel)
- Cloo NuGet package

MIT License • C# + OpenCL • Works on NVIDIA/AMD/Intel GPUs
