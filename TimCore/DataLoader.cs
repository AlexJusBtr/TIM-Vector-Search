using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
namespace TIMShield
{
    public static class DataLoader
    {
        public static List<string> LoadData(string filePath, bool isMalware)
        {
            var items = new List<string>();
            // Include '@' for email-like entries
            string validChars = isMalware ? null : "abcdefghijklmnopqrstuvwxyz0123456789-.@";
            try
            {
                if (!File.Exists(filePath))
                {
                    Console.WriteLine($"Error: File '{filePath}' not found.");
                    return items;
                }
                foreach (var line in File.ReadLines(filePath))
                {
                    if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
                    var parts = line.Split(',');
                    if (parts.Length < 1) continue;
                    string item = parts[0].Trim().ToLower();
                    if (isMalware || item.All(c => validChars.Contains(c)))
                    {
                        items.Add(item);
                    }
                    else
                    {
                        Console.WriteLine($"Warning: Skipped item due to invalid characters: {item.Substring(0, Math.Min(item.Length, 20))}...");
                    }
                }
            }
            catch (IOException ex)
            {
                Console.WriteLine($"Error reading file: {ex.Message}");
            }
            return items;
        }
    }
}