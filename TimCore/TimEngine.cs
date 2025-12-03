using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Cloo;
namespace TIMShield
{
    public class TemporalInterferenceMemory
    {
        private readonly int Dimensions;
        private readonly int MaxResonanceNodes;
        private readonly float ResonanceThreshold;
        private const float MinSimilarityThreshold = 0.7f;
        private readonly int MinSharedNodes;
        private readonly List<string> _items;
        private readonly List<TemporalPattern> _temporalMemory;
        private readonly Dictionary<int, List<int>> _resonanceIndex;
        private readonly float[] _baseFrequencies;
        private readonly float[] _phaseModulators;
        private readonly Random _random = new Random(42);
        private ComputeContext _context;
        private ComputeCommandQueue _queue;
        private ComputeProgram _program;
        private readonly bool _isMalware;
        private readonly bool _verbose;
        // Simple logging helpers
        private void Debug(string msg)
        {
            if (_verbose) Console.WriteLine(msg);
        }
        private void Info(string msg)
        {
            Console.WriteLine(msg);
        }
        // Add optional verbose flag (default false) to control console output
        public TemporalInterferenceMemory(List<string> items, bool isMalware, bool verbose = false)
        {
            _items = items ?? throw new ArgumentNullException(nameof(items));
            _isMalware = isMalware;
            _verbose = verbose;
            Dimensions = Math.Min(32, Math.Max(16, (int)Math.Log2(items.Count) * 2));
            MaxResonanceNodes = Math.Min(8, Math.Max(4, (int)Math.Log10(items.Count) + 2));
            ResonanceThreshold = items.Count > 10000 ? 0.75f : items.Count > 1000 ? 0.85f : 0.95f;
            MinSharedNodes = Math.Max(2, (int)Math.Log10(items.Count) + 1);
            _temporalMemory = new List<TemporalPattern>(items.Count);
            _resonanceIndex = new Dictionary<int, List<int>>(Dimensions);
            _baseFrequencies = new float[Dimensions];
            _phaseModulators = new float[Dimensions];
            for (int idx = 0; idx < Dimensions; idx++)
            {
                _baseFrequencies[idx] = (float)(_random.NextDouble() * 30 + 5);
                _phaseModulators[idx] = (float)(_random.NextDouble() * 2 * Math.PI);
                _resonanceIndex[idx] = new List<int>(Math.Max(1000, items.Count / 5));
            }
            InitializeOpenCL();
        }
        private void InitializeOpenCL()
        {
            if (ComputePlatform.Platforms.Count == 0)
            {
                throw new InvalidOperationException("No OpenCL platforms available.");
            }
            var platform = ComputePlatform.Platforms.FirstOrDefault(p => p.Name.Contains("NVIDIA"));
            if (platform == null)
            {
                platform = ComputePlatform.Platforms[0];
                Debug("Warning: No NVIDIA platform found, using default platform: " + platform.Name);
            }
            var devices = platform.Devices.Where(d => d.Type.HasFlag(ComputeDeviceTypes.Gpu)).ToList();
            if (!devices.Any())
            {
                devices = platform.Devices.Where(d => d.Type.HasFlag(ComputeDeviceTypes.Cpu)).ToList();
                Debug("Warning: No GPU found, falling back to CPU: " + devices.FirstOrDefault()?.Name);
            }
            if (!devices.Any())
            {
                throw new InvalidOperationException("No OpenCL devices available.");
            }
            try
            {
                var properties = new ComputeContextPropertyList(platform);
                _context = new ComputeContext(devices, properties, null, IntPtr.Zero);
                _queue = new ComputeCommandQueue(_context, devices[0], ComputeCommandQueueFlags.None);
                Info($"Using device: {devices[0].Name} (Type: {devices[0].Type}, MaxWorkGroupSize: {devices[0].MaxWorkGroupSize})");
                Debug($"Max Mem Alloc Size: {devices[0].MaxMemoryAllocationSize / (1024 * 1024)} MB, Global Mem Size: {devices[0].GlobalMemorySize / (1024 * 1024)} MB");
            }
            catch (ComputeException ex)
            {
                Debug($"Failed to initialize OpenCL context: {ex.Message}, Code: {ex}");
                throw;
            }
            string kernelSource = GetOpenClKernelSource();
            _program = new ComputeProgram(_context, kernelSource);
            try
            {
                _program.Build(null, null, null, IntPtr.Zero);
                Info("OpenCL program built successfully");
            }
            catch (ComputeException ex)
            {
                string buildLog = string.Join("\n", _context.Devices.Select(d => $"Device {d.Name}: {_program.GetBuildLog(d)}"));
                Debug($"OpenCL build error details:\n{buildLog}");
                throw;
            }
        }
        public void BuildTemporalMemory()
        {
            if (!_items.Any())
            {
                throw new InvalidOperationException("No items provided.");
            }
            var stopwatch = Stopwatch.StartNew();
            int maxLength = _items.Max(s => s.Length);
            var inputItems = _items.Select(s => s.PadRight(maxLength, '\0')).ToArray();
            var patterns = _items.Count < 100 ? EncodeToTemporalPatternsCpu(inputItems, maxLength) : EncodeToTemporalPatternsGpu(inputItems, maxLength);
            for (int idx = 0; idx < patterns.Length; idx++)
            {
                _temporalMemory.Add(patterns[idx]);
                foreach (var node in patterns[idx].ResonanceNodes)
                {
                    _resonanceIndex[node].Add(idx);
                }
            }
            int targetIndex = _items.IndexOf(_isMalware ? "a1b2c3d4e5f6" : "paypa1.com");
            if (targetIndex >= 0)
            {
                float magnitude = (float)Math.Sqrt(_temporalMemory[targetIndex].WaveVector.Sum(x => x * x));
                Debug($"Dataset nodes for {_items[targetIndex]} (index {targetIndex}): {string.Join(", ", _temporalMemory[targetIndex].ResonanceNodes.Select(n => n.ToString()))}");
                Debug($"Dataset WaveVector (first 5): {string.Join(", ", _temporalMemory[targetIndex].WaveVector.Take(5).Select(f => f.ToString("F6")))}...");
                Debug($"WaveVector magnitude: {magnitude:F6}");
                if (magnitude < 0.0001f)
                {
                    Debug($"Warning: Invalid wave vector for {_items[targetIndex]}");
                }
            }
            foreach (var list in _resonanceIndex.Values)
            {
                list.TrimExcess();
            }
            _temporalMemory.TrimExcess();
            stopwatch.Stop();
            Info($"Temporal memory built: {_temporalMemory.Count:N0} patterns in {stopwatch.ElapsedMilliseconds:F0} ms");
            Debug($"Resonance nodes: {_resonanceIndex.Values.Sum(l => l.Count):N0}");
            Debug($"Resonance node distribution: {string.Join(", ", _resonanceIndex.Select(kv => $"{kv.Key}:{kv.Value.Count}"))}");
        }
        private TemporalPattern[] EncodeToTemporalPatternsGpu(string[] inputItems, int maxLength)
{
    int numItems = inputItems.Length;
    Console.WriteLine($"Encoding {numItems} items (GPU)");
    var flatInputBytes = Encoding.ASCII.GetBytes(string.Concat(inputItems));
    Console.WriteLine($"Flat input length: {flatInputBytes.Length}");
    // Validate buffer sizes
    long inputBufferSize = flatInputBytes.Length;
    long waveVectorsSize = (long)numItems * Dimensions * sizeof(float);
    long resonanceNodesSize = (long)numItems * MaxResonanceNodes * sizeof(int);
    long nodeCountsSize = numItems * sizeof(int);
    long maxMemAlloc = _context.Devices[0].MaxMemoryAllocationSize;
    if (waveVectorsSize > maxMemAlloc || resonanceNodesSize > maxMemAlloc || nodeCountsSize > maxMemAlloc)
    {
        Console.WriteLine($"Error: Buffer size exceeds device limit ({maxMemAlloc / (1024 * 1024)} MB)");
        throw new InvalidOperationException("Buffer size exceeds GPU memory limit.");
    }
    ComputeBuffer<byte> inputBuffer = null;
    ComputeBuffer<float> waveVectorsBuffer = null;
    ComputeBuffer<int> resonanceNodesBuffer = null;
    ComputeBuffer<int> nodeCountsBuffer = null;
    ComputeBuffer<float> frequenciesBuffer = null;
    ComputeBuffer<float> modulatorsBuffer = null;
    ComputeKernel kernel = null;
    try
    {
        inputBuffer = new ComputeBuffer<byte>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, flatInputBytes);
        waveVectorsBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.WriteOnly, numItems * Dimensions);
        resonanceNodesBuffer = new ComputeBuffer<int>(_context, ComputeMemoryFlags.WriteOnly, numItems * MaxResonanceNodes);
        nodeCountsBuffer = new ComputeBuffer<int>(_context, ComputeMemoryFlags.WriteOnly, numItems);
        frequenciesBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, _baseFrequencies);
        modulatorsBuffer = new ComputeBuffer<float>(_context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, _phaseModulators);
        kernel = _program.CreateKernel("encodeKernel");
        kernel.SetMemoryArgument(0, inputBuffer);
        kernel.SetMemoryArgument(1, waveVectorsBuffer);
        kernel.SetMemoryArgument(2, resonanceNodesBuffer);
        kernel.SetMemoryArgument(3, nodeCountsBuffer);
        kernel.SetMemoryArgument(4, frequenciesBuffer);
        kernel.SetMemoryArgument(5, modulatorsBuffer);
        kernel.SetValueArgument(6, numItems);
        kernel.SetValueArgument(7, maxLength);
        kernel.SetValueArgument(8, Dimensions);
        kernel.SetValueArgument(9, MaxResonanceNodes);
        kernel.SetValueArgument(10, ResonanceThreshold);
        long maxWorkItems = Math.Min(_context.Devices[0].MaxWorkGroupSize, numItems);
        for (long offset = 0; offset < numItems; offset += maxWorkItems)
        {
            long batchSize = Math.Min(maxWorkItems, numItems - offset);
            _queue.Execute(kernel, new long[] { offset }, new long[] { batchSize }, null, null);
        }
        _queue.Finish(); // Ensure all commands complete
        Console.WriteLine("OpenCL kernel executed successfully");
        float[] waveVectors = new float[numItems * Dimensions];
        int[] resonanceNodes = new int[numItems * MaxResonanceNodes];
        int[] nodeCounts = new int[numItems];
        try
        {
            Debug($"Reading waveVectorsBuffer (size: {waveVectors.Length * sizeof(float)} bytes)");
            _queue.ReadFromBuffer(waveVectorsBuffer, ref waveVectors, true, null);
            Debug("Successfully read waveVectorsBuffer");
            Debug($"Reading resonanceNodesBuffer (size: {resonanceNodes.Length * sizeof(int)} bytes)");
            _queue.ReadFromBuffer(resonanceNodesBuffer, ref resonanceNodes, true, null);
            Debug("Successfully read resonanceNodesBuffer");
            Debug($"Reading nodeCountsBuffer (size: {nodeCounts.Length * sizeof(int)} bytes)");
            _queue.ReadFromBuffer(nodeCountsBuffer, ref nodeCounts, true, null);
            Debug("Successfully read nodeCountsBuffer");
        }
        catch (ComputeException ex)
        {
            Debug($"OpenCL read error: {ex.Message}, Code: {ex}");
            throw;
        }
        var patterns = new TemporalPattern[numItems];
        for (int idx = 0; idx < numItems; idx++)
        {
            var waveVec = new float[Dimensions];
            Array.Copy(waveVectors, idx * Dimensions, waveVec, 0, Dimensions);
            var nodes = new List<int>();
            for (int j = 0; j < nodeCounts[idx] && j < MaxResonanceNodes; j++)
            {
                int node = resonanceNodes[idx * MaxResonanceNodes + j];
                if (node >= 0 && node < Dimensions)
                {
                    nodes.Add(node);
                }
            }
            patterns[idx] = new TemporalPattern
            {
                WaveVector = waveVec,
                ResonanceNodes = nodes
            };
            // Log wave vector for debugging
            float magnitude = (float)Math.Sqrt(waveVec.Sum(x => x * x));
            if (magnitude < 0.0001f)
            {
                Debug($"Warning: Zero magnitude wave vector for item {idx} ({inputItems[idx]})");
            }
            else
            {
                Debug($"Wave vector for {inputItems[idx]} (first 5): {string.Join(", ", waveVec.Take(5).Select(f => f.ToString("F6")))}...");
            }
        }
        return patterns;
    }
    catch (ComputeException ex)
    {
        Console.WriteLine($"OpenCL error: {ex.Message}, Code: {ex}");
        throw;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Unexpected error: {ex.GetType().Name}: {ex.Message}");
        throw;
    }
    finally
    {
        inputBuffer?.Dispose();
        waveVectorsBuffer?.Dispose();
        resonanceNodesBuffer?.Dispose();
        nodeCountsBuffer?.Dispose();
        frequenciesBuffer?.Dispose();
        modulatorsBuffer?.Dispose();
        kernel?.Dispose();
    }
}
        private TemporalPattern[] EncodeToTemporalPatternsCpu(string[] inputItems, int maxLength)
        {
            int numItems = inputItems.Length;
            Debug($"Encoding {numItems} items (CPU)");
            var patterns = new TemporalPattern[numItems];
            for (int idx = 0; idx < numItems; idx++)
            {
                var waveVector = new float[Dimensions];
                var nodes = new List<int>();
                int nodeCount = 0;
                int validChars = 0;
                string input = inputItems[idx];
                for (int pos = 0; pos < maxLength && input[pos] != '\0'; pos++)
                {
                    char c = input[pos];
                    float charValue = _isMalware ? (float)c : -1.0f;
                    if (!_isMalware)
                    {
                        // Include '@' in the alphabet for domain/email entries
                        string alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-.@";
                        int alphabetIndex = alphabet.IndexOf(c);
                        if (alphabetIndex >= 0)
                        {
                            charValue = alphabetIndex;
                        }
                    }
                    if (charValue < 0.0f) continue;
                    validChars++;
                    float charFrequency = charValue * (_isMalware ? 0.1f : 0.2f);
                    float positionPhase = pos * (float)Math.PI / maxLength;
                    float lengthScale = maxLength < 50 ? 1.5f : 1.0f;
                    float posScale = (1.0f + pos * 0.05f) * lengthScale;
                    for (int d = 0; d < Dimensions; d++)
                    {
                        float wave = (float)Math.Sin(_baseFrequencies[d] * charFrequency + _phaseModulators[d] + positionPhase);
                        waveVector[d] += wave * posScale;
                        if (Math.Abs(wave) > ResonanceThreshold && nodeCount < MaxResonanceNodes)
                        {
                            nodes.Add(d);
                            nodeCount++;
                        }
                    }
                }
                if (validChars == 0)
                {
                    patterns[idx] = new TemporalPattern { WaveVector = new float[Dimensions], ResonanceNodes = new List<int>() };
                    continue;
                }
                float magnitude = 0.0f;
                for (int d = 0; d < Dimensions; d++)
                {
                    magnitude += waveVector[d] * waveVector[d];
                }
                magnitude = (float)Math.Sqrt(magnitude);
                if (magnitude < 0.0001f)
                {
                    patterns[idx] = new TemporalPattern { WaveVector = new float[Dimensions], ResonanceNodes = new List<int>() };
                    continue;
                }
                for (int d = 0; d < Dimensions; d++)
                {
                    waveVector[d] /= magnitude;
                }
                patterns[idx] = new TemporalPattern { WaveVector = waveVector, ResonanceNodes = nodes };
            }
            return patterns;
        }
        private string GetOpenClKernelSource()
        {
            return _isMalware ? $@"
                __kernel void encodeKernel(
                    __global const char* input,
                    __global float* waveVectors,
                    __global int* resonanceNodes,
                    __global int* nodeCounts,
                    __global const float* baseFrequencies,
                    __global const float* phaseModulators,
                    int itemCount,
                    int maxLength,
                    int Dimensions,
                    int MaxResonanceNodes,
                    float ResonanceThreshold)
                {{
                    int idx = get_global_id(0);
                    if (idx >= itemCount) return;
                    float waveVector[32];
                    int nodes[8];
                    for (int d = 0; d < Dimensions; d++) waveVector[d] = 0.0f;
                    for (int n = 0; n < MaxResonanceNodes; n++) nodes[n] = -1;
                    int nodeCount = 0;
                    int validChars = 0;
                    for (int pos = 0; pos < maxLength; pos++)
                    {{
                        char c = input[idx * maxLength + pos];
                        if (c == '\0') break;
                        float charValue = (float)c;
                        validChars++;
                        float charFrequency = charValue * 0.1f;
                        float positionPhase = pos * 3.1415926f / maxLength;
                        float lengthScale = maxLength < 50 ? 1.5f : 1.0f;
                        float posScale = (1.0f + pos * 0.05f) * lengthScale;
                        for (int d = 0; d < Dimensions; d++)
                        {{
                            float wave = sin(baseFrequencies[d] * charFrequency + phaseModulators[d] + positionPhase);
                            waveVector[d] += wave * posScale;
                            if (fabs(wave) > ResonanceThreshold && nodeCount < MaxResonanceNodes)
                            {{
                                nodes[nodeCount++] = d;
                            }}
                        }}
                    }}
                    if (validChars == 0)
                    {{
                        for (int d = 0; d < Dimensions; d++) waveVectors[idx * Dimensions + d] = 0.0f;
                        for (int n = 0; n < MaxResonanceNodes; n++) resonanceNodes[idx * MaxResonanceNodes + n] = -1;
                        nodeCounts[idx] = 0;
                        return;
                    }}
                    float magnitude = 0.0f;
                    for (int d = 0; d < Dimensions; d++)
                        magnitude += waveVector[d] * waveVector[d];
                    magnitude = sqrt(magnitude);
                    if (magnitude < 0.0001f)
                    {{
                        for (int d = 0; d < Dimensions; d++) waveVectors[idx * Dimensions + d] = 0.0f;
                        for (int n = 0; n < MaxResonanceNodes; n++) resonanceNodes[idx * MaxResonanceNodes + n] = -1;
                        nodeCounts[idx] = 0;
                        return;
                    }}
                    for (int d = 0; d < Dimensions; d++)
                        waveVector[d] /= magnitude;
                    for (int d = 0; d < Dimensions; d++)
                        waveVectors[idx * Dimensions + d] = waveVector[d];
                    for (int n = 0; n < MaxResonanceNodes; n++) resonanceNodes[idx * MaxResonanceNodes + n] = nodes[n];
                    nodeCounts[idx] = nodeCount;
                }}"
                : $@"
                __kernel void encodeKernel(
    __global const char* input,
    __global float* waveVectors,
    __global int* resonanceNodes,
    __global int* nodeCounts,
    __global const float* baseFrequencies,
    __global const float* phaseModulators,
    int itemCount,
    int maxLength,
    int Dimensions,
    int MaxResonanceNodes,
    float ResonanceThreshold)
{{
    int idx = get_global_id(0);
    if (idx >= itemCount) return;
    float waveVector[32];
    int nodes[8];
    for (int d = 0; d < Dimensions; d++) waveVector[d] = 0.0f;
    for (int n = 0; n < MaxResonanceNodes; n++) nodes[n] = -1;
    int nodeCount = 0;
    int validChars = 0;
    __constant char alphabet[] = ""abcdefghijklmnopqrstuvwxyz0123456789-.@"";
    const int alphabetSize = 39;
    int inputOffset = idx * maxLength;
    for (int pos = 0; pos < maxLength; pos++)
    {{
        char c = input[inputOffset + pos];
        if (c == '\0') break;
        float charValue = -1.0f;
        for (int i = 0; i < alphabetSize; i++)
        {{
            if (c == alphabet[i])
            {{
                charValue = (float)i;
                break;
            }}
        }}
        if (charValue < 0.0f) continue; // Skip invalid characters
        validChars++;
        float charFrequency = charValue * 0.2f;
        float positionPhase = pos * 3.1415926f / maxLength;
        float lengthScale = maxLength < 50 ? 1.5f : 1.0f;
        float posScale = (1.0f + pos * 0.05f) * lengthScale;
        for (int d = 0; d < Dimensions; d++)
        {{
            float wave = sin(baseFrequencies[d] * charFrequency + phaseModulators[d] + positionPhase);
            waveVector[d] += wave * posScale;
            if (fabs(wave) > ResonanceThreshold && nodeCount < MaxResonanceNodes)
            {{
                nodes[nodeCount++] = d;
            }}
        }}
    }}
    if (validChars == 0)
    {{
        for (int d = 0; d < Dimensions; d++)
            waveVectors[idx * Dimensions + d] = 0.0f;
        for (int n = 0; n < MaxResonanceNodes; n++)
            resonanceNodes[idx * MaxResonanceNodes + n] = -1;
        nodeCounts[idx] = 0;
        return;
    }}
    float magnitude = 0.0f;
    for (int d = 0; d < Dimensions; d++)
    {{
        magnitude += waveVector[d] * waveVector[d];
    }}
    magnitude = sqrt(magnitude);
    if (magnitude < 0.0001f)
    {{
        for (int d = 0; d < Dimensions; d++)
            waveVectors[idx * Dimensions + d] = 0.0f;
        for (int n = 0; n < MaxResonanceNodes; n++)
            resonanceNodes[idx * MaxResonanceNodes + n] = -1;
        nodeCounts[idx] = 0;
        return;
    }}
   
    for (int d = 0; d < Dimensions; d++)
    {{
        waveVector[d] /= magnitude;
    }}
    for (int d = 0; d < Dimensions; d++)
    {{
        waveVectors[idx * Dimensions + d] = waveVector[d];
    }}
    for (int n = 0; n < MaxResonanceNodes; n++)
    {{
        resonanceNodes[idx * MaxResonanceNodes + n] = nodes[n];
    }}
    nodeCounts[idx] = nodeCount;
}}";
        }
        public List<TemporalSearchResult> SearchWithTemporalInterference(string query, int topK)
        {
            var stopwatch = Stopwatch.StartNew();
            int maxLength = _items.Max(s => s.Length);
            // Preprocess query to remove invalid characters
            string validChars = "abcdefghijklmnopqrstuvwxyz0123456789-.@";
            string cleanedQuery = new string(query.ToLower().Where(c => validChars.Contains(c)).ToArray());
            Info($"Cleaned query: {cleanedQuery}");
            var queryPattern = _items.Count < 100 ? EncodeToTemporalPatternCpu(cleanedQuery, maxLength) : EncodeToTemporalPatternGpu(cleanedQuery, maxLength);
            var candidateScores = new Dictionary<int, (float Score, float Fidelity, float PhaseDifference)>(Math.Max(50, _items.Count / 10));
            var candidatesExplored = 0;
            var results = new List<TemporalSearchResult>();
            Debug($"Query nodes for {query}: {string.Join(", ", queryPattern.ResonanceNodes.Select(n => n.ToString()))}");
            Debug($"Query WaveVector (first 5): {string.Join(", ", queryPattern.WaveVector.Take(5).Select(f => f.ToString("F6")))}...");
            var candidateSet = new HashSet<int>(Math.Max(50, _items.Count / 10));
            var nodeCount = new Dictionary<int, int>();
            foreach (var node in queryPattern.ResonanceNodes)
            {
                if (_resonanceIndex.ContainsKey(node))
                {
                    foreach (var idx in _resonanceIndex[node])
                    {
                        nodeCount[idx] = nodeCount.GetValueOrDefault(idx, 0) + 1;
                        if (nodeCount[idx] >= MinSharedNodes)
                        {
                            candidateSet.Add(idx);
                        }
                    }
                }
            }
            int targetIndex = _items.IndexOf(query);
            if (targetIndex >= 0)
            {
                candidateSet.Add(targetIndex);
            }
            double pruningPercentage = (double)candidateSet.Count / _items.Count * 100;
            Info($"Candidate set size: {candidateSet.Count} (contains query index {targetIndex}: {candidateSet.Contains(targetIndex)})");
            Info($"Database searched: {candidateSet.Count:N0} items ({pruningPercentage:F2}% of total {_items.Count:N0} items)");
            Info($"Pruning efficiency: {(100 - pruningPercentage):F2}%");
            var topCandidates = new List<(float Score, int Index, float Fidelity, float PhaseDifference)>();
            foreach (var idx in candidateSet)
            {
                var candidatePattern = _temporalMemory[idx];
                var (similarity, fidelity, phaseDifference) = CalculateWaveSimilarity(queryPattern, candidatePattern);
                Debug($"Comparing {query} with {_items[idx]}: Fidelity={fidelity:F4}, PhaseDiff={phaseDifference:F4}, Score={similarity:F4}");
                if (similarity >= MinSimilarityThreshold || idx == targetIndex)
                {
                    candidatesExplored++;
                    candidateScores[idx] = (similarity, fidelity, phaseDifference);
                    topCandidates.Add((similarity, idx, fidelity, phaseDifference));
                }
            }
            topCandidates.Sort((a, b) => b.Score.CompareTo(a.Score));
            topCandidates = topCandidates.Take(topK).ToList();
            foreach (var candidate in topCandidates)
            {
                results.Add(new TemporalSearchResult
                {
                    FoundItem = _items[candidate.Index],
                    IsExactMatch = _items[candidate.Index] == query,
                    ResonanceStrength = candidate.Score,
                    Fidelity = candidate.Fidelity * 100,
                    PhaseDifference = candidate.PhaseDifference,
                    SearchTime = stopwatch.Elapsed,
                    CandidatesExplored = candidatesExplored
                });
            }
            // Summary: pruning and exploration stats
            Info($"Candidates explored (passed similarity threshold): {candidatesExplored}");
            Info($"Total candidates considered: {candidateSet.Count}");
            stopwatch.Stop();
            return results;
        }
        private TemporalPattern EncodeToTemporalPatternGpu(string input, int maxLength)
        {
            var inputItems = new[] { input.ToLower().PadRight(maxLength, '\0') };
            return EncodeToTemporalPatternsGpu(inputItems, maxLength)[0];
        }
        private TemporalPattern EncodeToTemporalPatternCpu(string input, int maxLength)
        {
            var inputItems = new[] { input.ToLower().PadRight(maxLength, '\0') };
            return EncodeToTemporalPatternsCpu(inputItems, maxLength)[0];
        }
        private (float Similarity, float Fidelity, float PhaseDifference) CalculateWaveSimilarity(TemporalPattern target, TemporalPattern candidate)
        {
            float targetMagnitude = 0, candidateMagnitude = 0;
            for (int d = 0; d < Dimensions; d++)
            {
                targetMagnitude += target.WaveVector[d] * target.WaveVector[d];
                candidateMagnitude += candidate.WaveVector[d] * candidate.WaveVector[d];
            }
            targetMagnitude = (float)Math.Sqrt(targetMagnitude);
            candidateMagnitude = (float)Math.Sqrt(candidateMagnitude);
            if (targetMagnitude < 0.0001f || candidateMagnitude < 0.0001f)
            {
                return (0, 0, (float)Math.PI);
            }
            float dotProduct = 0;
            for (int d = 0; d < Dimensions; d++)
            {
                dotProduct += target.WaveVector[d] * candidate.WaveVector[d];
            }
            float cosineSimilarity = dotProduct / (targetMagnitude * candidateMagnitude);
            cosineSimilarity = Math.Max(-1, Math.Min(1, cosineSimilarity));
            float fidelity = cosineSimilarity * cosineSimilarity;
            float phaseDifference = (float)Math.Acos(cosineSimilarity);
            float phasePenalty = phaseDifference / (float)Math.PI * 0.1f; // Reduced penalty
            int sharedNodes = target.ResonanceNodes.Intersect(candidate.ResonanceNodes).Count();
            float nodeBonus = sharedNodes * 0.1f; // Reduced node bonus
            float similarity = Math.Max(0, fidelity - phasePenalty + nodeBonus);
            return (similarity, fidelity, phaseDifference);
        }
    }
}