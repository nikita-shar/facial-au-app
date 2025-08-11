using UnityEngine;
using TMPro;
using System.Linq;

public class TextAULabels : MonoBehaviour
{
    [Header("Tracker here")]
    public LiveFaceTracker tracker;

    [Header("5 TextMesh objects")]
    public TextMeshProUGUI au1Label;
    public TextMeshProUGUI au7Label;
    public TextMeshProUGUI au10Label;
    public TextMeshProUGUI au12Label;
    public TextMeshProUGUI au17Label;

    void Update()
    {
        var smoothed = tracker.LatestSmoothedAUs;
        Debug.Log($"[TextAULabels] LatestSmoothedAUs = {(smoothed == null ? "null" : string.Join(", ", smoothed.Take(5).Select(v=>v.ToString("F4"))))}");
        if (smoothed == null || smoothed.Length < 12) return;

        // Update text (format each to two decimals)
        au1Label.text  = smoothed[0].ToString("F4");  
        au7Label.text  = smoothed[4].ToString("F4");   
        au10Label.text  = smoothed[5].ToString("F4"); 
        au12Label.text = smoothed[6].ToString("F4");   
        au17Label.text = smoothed[9].ToString("F4");  
    }
}
