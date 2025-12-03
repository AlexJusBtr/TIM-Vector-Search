using System;
using System.IO; // Added for Path and File checks
using System.Linq;

namespace TIMShield
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("==TIMShield: Temporal Interference Memory Search for Cybersecurity==");
            Console.WriteLine("Enter data type (Domain/Malware):");
            string dataType = Console.ReadLine()?.Trim().ToUpper();
            bool isMalware = dataType == "MALWARE";

            Console.WriteLine("Press a key to start...");
            Console.ReadKey();

            // --- FIXED SECTION START ---
            
            // 1. Determine the file name based on user choice
            string fileName = isMalware ? "malware_signatures.csv" : "domain_list.csv";
            
            // 2. Build the path relative to where the application is running
            string dataFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, fileName);

            // 3. Debug check: Print where it is looking (Optional, helps when testing)
            // Console.WriteLine($"Looking for file at: {dataFilePath}");

            if (!File.Exists(dataFilePath))
            {
                Console.WriteLine($"\nCRITICAL ERROR: Could not find '{fileName}'.");
                Console.WriteLine($"Make sure 'Copy to Output Directory' is set to 'Copy if newer' in the file properties.");
                Console.WriteLine("Press a key to exit...");
                Console.ReadKey();
                return;
            }
            // --- FIXED SECTION END ---

            var dataItems = DataLoader.LoadData(dataFilePath, isMalware);

            if (dataItems.Count == 0)
            {
                Console.WriteLine("Error: No valid data found in the file. Please check the file format and content.");
                Console.WriteLine("Press a key to exit...");
                Console.ReadKey();
                return;
            }

            Console.WriteLine($"\nLoaded {dataItems.Count:N0} {(isMalware ? "malware signatures" : "domains")} from file.");

            // Example query
            string query = isMalware ? "a1b2c3d4e5f6" : "p@ypal.com";
            Console.WriteLine($"Query: {query}");
            Console.WriteLine($"Query length: {query.Length}, Search space: {Math.Pow(isMalware ? 256 : 36, query.Length):E2} combinations");

            // Initialize and build temporal memory
            var temporalMemory = new TemporalInterferenceMemory(dataItems, isMalware, verbose: false);
            try
            {
                temporalMemory.BuildTemporalMemory();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error building temporal memory: {ex.Message}");
                Console.WriteLine("Press a key to exit...");
                Console.ReadKey();
                return;
            }

            // Perform search
            Console.WriteLine("\nInitiating TIMShield Search...");
            var results = temporalMemory.SearchWithTemporalInterference(query, topK: 5);

            // Display results
            Console.WriteLine("\n== Search Results ==");
            foreach (var result in results)
            {
                Console.WriteLine($"Match: {result.FoundItem}");
                Console.WriteLine($"Fidelity: {result.Fidelity:F2}%");
                Console.WriteLine($"Phase Difference: {result.PhaseDifference:F2} radians");
                Console.WriteLine($"Exact match: {result.IsExactMatch}");
                Console.WriteLine($"Resonance strength: {result.ResonanceStrength:F6}");
                Console.WriteLine($"Search time: {result.SearchTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"Candidates explored (for this result): {result.CandidatesExplored}");
                Console.WriteLine("---");
            }

            Console.WriteLine("Press a key to exit...");
            Console.ReadKey();
        }
    }
}