using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;


public class ButtonManager : MonoBehaviour
{
    public GameObject baseFacePromptText;
    public Button baseStartButton;
    public Button startButton;
    public Button stopButton;
    public Button leftButton;
    public Button rightButton;
    private string selectedSide = "";
    private bool hasStarted = false;

    private Button selectedTrackingButton;
    private Button selectedSideButton;

    public LiveFaceTracker tracker;

    private Color normalColor = Color.white;
    private Color activeColor = new Color(0.2f, 0.6f, 1f); // Light blue

    void Start()
    {
        //record reference face first 
        baseFacePromptText.SetActive(true);
        baseStartButton.gameObject.SetActive(true);
        startButton.gameObject.SetActive(false);
        stopButton.gameObject.SetActive(false);
        leftButton.gameObject.SetActive(false);
        rightButton.gameObject.SetActive(false);

        baseStartButton.onClick.AddListener(() => StartCoroutine(tracker.RecordNeutralFaceCoroutine(AfterNeutralFaceCaptured)));


        startButton.onClick.AddListener(StartTracking);
        stopButton.onClick.AddListener(StopTracking);

        leftButton.onClick.AddListener(() => SelectSide("Left"));
        rightButton.onClick.AddListener(() => SelectSide("Right"));


        // Set initial button colors
        SetButtonColors(startButton, false);
        SetButtonColors(stopButton, false);

        tracker.isTrackingActive = false;
    }

    // void StartTracking()
    // {
    //     tracker.isTrackingActive = true;
    //     Debug.Log("Tracking started.");
    //     SetButtonColors(startButton, true);
    //     SetButtonColors(stopButton, false);
    // }

    // void StartTracking()
    // {
    //     if (string.IsNullOrEmpty(selectedSide))
    //     {
    //         Debug.LogWarning("Please select Left or Right side before starting tracking.");
    //         return;
    //     }

    //     tracker.isTrackingActive = true;
    //     Debug.Log("Start button clicked — waiting for side selection.");
    //     selectedTrackingButton = startButton;
    //     SetButtonColors(startButton, true);
    //     SetButtonColors(stopButton, false);
    // }

    void StartTracking()
    {
        if (string.IsNullOrEmpty(selectedSide))
        {
            Debug.LogWarning("Please select Left or Right side before starting tracking.");
            return;
        }

        tracker.isTrackingActive = true;
        hasStarted = true;
        Debug.Log("Tracking started on " + selectedSide + " side.");
        selectedTrackingButton = startButton;
        SetButtonColors(startButton, true);
        SetButtonColors(stopButton, false);
    }




    void StopTracking()
    {
        if (!hasStarted)
        {
            Debug.LogWarning("You must press Start before Stop.");
            return;
        }

        tracker.isTrackingActive = false;
        Debug.Log("Tracking stopped.");
        selectedTrackingButton = stopButton;
        SetButtonColors(startButton, false);
        SetButtonColors(stopButton, true);

        //tracker.FinalizeCSVOnStop();

    }


    // void SetButtonColors(Button button, bool isActive)
    // {
    //     ColorBlock cb = button.colors;
    //     cb.normalColor = isActive ? activeColor : normalColor;
    //     cb.selectedColor = isActive ? activeColor : normalColor;
    //     cb.highlightedColor = isActive ? activeColor : normalColor;
    //     cb.pressedColor = isActive ? activeColor : normalColor;
    //     cb.disabledColor = new Color(0.7f, 0.7f, 0.7f); // optional
    //     button.colors = cb;
    // }

    void SetButtonColors(Button button, bool isActive)
    {
        ColorBlock cb = button.colors;
        cb.normalColor = isActive ? activeColor : normalColor;
        cb.highlightedColor = cb.normalColor;
        cb.selectedColor = cb.normalColor;
        cb.pressedColor = cb.normalColor;
        cb.disabledColor = new Color(0.7f, 0.7f, 0.7f);
        button.colors = cb;
    }



    // private void SelectSide(string side)
    // {
    //     selectedSide = side;

    //     ColorBlock leftColors = leftButton.colors;
    //     ColorBlock rightColors = rightButton.colors;

    //     if (side == "Left")
    //     {
    //         leftColors.normalColor = Color.blue;
    //         rightColors.normalColor = Color.white;
    //     }
    //     else
    //     {
    //         leftColors.normalColor = Color.white;
    //         rightColors.normalColor = Color.blue;
    //     }

    //     leftButton.colors = leftColors;
    //     rightButton.colors = rightColors;
    // }


    private void SelectSide(string side)
    {
        selectedSide = side;
        Debug.Log("Selected side: " + selectedSide);

        selectedSideButton = (side == "Left") ? leftButton : rightButton;

        SetButtonColors(leftButton, side == "Left");
        SetButtonColors(rightButton, side == "Right");
    }

    public bool SelectedSideIsSet()
    {
        return !string.IsNullOrEmpty(selectedSide);
    }
    private void AfterNeutralFaceCaptured()
    {
        baseFacePromptText.SetActive(false);
        baseStartButton.gameObject.SetActive(false);

        startButton.gameObject.SetActive(true);
        stopButton.gameObject.SetActive(true);
        leftButton.gameObject.SetActive(true);
        rightButton.gameObject.SetActive(true);
    }

    public void HideBaseUIAndShowTrackingUI()
    {
        if (baseFacePromptText != null)
            baseFacePromptText.SetActive(false);

        if (baseStartButton != null)
            baseStartButton.gameObject.SetActive(false);

        startButton.gameObject.SetActive(true);
        stopButton.gameObject.SetActive(true);
        leftButton.gameObject.SetActive(true);
        rightButton.gameObject.SetActive(true);
    }
    
    public bool IsLeftSelected  => selectedSide == "Left";
    public bool IsRightSelected => selectedSide == "Right";



}


