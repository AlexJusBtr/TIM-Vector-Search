using System;
using System.Collections.Generic;
namespace TIMShield
{
    public class TemporalPattern
    {
        public float[] WaveVector { get; set; }
        public List<int> ResonanceNodes { get; set; }
    }
    public class TemporalSearchResult
    {
        public string FoundItem { get; set; }
        public bool IsExactMatch { get; set; }
        public float ResonanceStrength { get; set; }
        public float Fidelity { get; set; }
        public float PhaseDifference { get; set; }
        public TimeSpan SearchTime { get; set; }
        public int CandidatesExplored { get; set; }
    }
}