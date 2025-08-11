using UnityEngine;
using UnityEngine.UI;

public class ToggleManager : MonoBehaviour
{
    [Header("Side Selection")]
    public Toggle leftToggle;
    public Toggle rightToggle;
    private string selectedSide = "";

    [Header("Tracking")]
    public Toggle startToggle;
    public Toggle stopToggle;

    public LiveFaceTracker tracker;

    void Start()
    {
        // Side toggles
        leftToggle.onValueChanged.AddListener((isOn) => { if (isOn) SelectSide("Left"); });
        rightToggle.onValueChanged.AddListener((isOn) => { if (isOn) SelectSide("Right"); });

        // Start/Stop toggles
        startToggle.onValueChanged.AddListener((isOn) => { if (isOn) StartTracking(); });
        stopToggle.onValueChanged.AddListener((isOn) => { if (isOn) StopTracking(); });

        // Set defaults
        tracker.isTrackingActive = false;
    }

    private void SelectSide(string side)
    {
        selectedSide = side;
        Debug.Log("Selected side: " + selectedSide);
    }

    private void StartTracking()
    {
        if (string.IsNullOrEmpty(selectedSide))
        {
            Debug.LogWarning("Please select Left or Right side before starting tracking.");
            stopToggle.isOn = true; // reset toggle
            return;
        }

        tracker.isTrackingActive = true;
        Debug.Log("Tracking started.");
    }

    private void StopTracking()
    {
        tracker.isTrackingActive = false;
        Debug.Log("Tracking stopped.");
    }
}
