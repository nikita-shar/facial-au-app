using System.Linq;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using TMPro;

//[RequireComponent(typeof(ARFaceManager))]
public class TextLabelCoord : MonoBehaviour
{
    public ARFaceManager faceManager;       
    public Camera mainCamera;     
    public LiveFaceTracker faceTracker; 
    public ButtonManager buttonManager;
    //public Canvas canvas;


    public TextMeshProUGUI[] auLabels;      

    //public int[] auIndices = { 0, 1, 4, 5, 9};
   
    //public int[] leftVertexIndices = {168, 164, 54, 238, 271};

    //public int[] rightVertexIndices = {617, 613, 504, 672, 706};

    public int[] auIndices = {0, 4, 5, 6, 9};
   
    public int[] leftVertexIndices = {421, 1153, 70, 174, 345};

    public int[] rightVertexIndices = {851, 1127, 519, 623, 778};


    void Update()
    {
        var sm = faceTracker.LatestSmoothedAUs;
        if (sm == null || sm.Length < 12) return;

        // take first face
        ARFace face = null;
        foreach (var f in faceManager.trackables) {
            face = f;
            break;
        }
        if (face == null) return;

        // local verts
        var verts = face.vertices.ToArray();

        // choose which side indices
        bool leftParalyzed = buttonManager.IsLeftSelected; 

        var vIndices = leftParalyzed ? leftVertexIndices : rightVertexIndices;

        for (int i = 0; i < auLabels.Length; i++)
        {
            float val = sm[auIndices[i]];
            auLabels[i].text = val.ToString("F2");
            auLabels[i].color = new Color(0f, 0.5f, 1f);

            var localPt = verts[vIndices[i]];
            var worldPt = face.transform.TransformPoint(localPt);
            var screenPt = mainCamera.WorldToScreenPoint(worldPt);
            auLabels[i].rectTransform.position = screenPt;
            // var screenPt = mainCamera.WorldToScreenPoint(worldPt);

            // // Convert the screen‐space point into the Canvas's local space
            // RectTransformUtility.ScreenPointToLocalPointInRectangle(
            //     canvas.transform as RectTransform,
            //     screenPt,
            //     canvas.renderMode == RenderMode.ScreenSpaceOverlay 
            //     ? null               // no camera needed in Overlay mode
            //     : canvas.worldCamera,
            //     out Vector2 canvasPos);

            // Place the label by its anchoredPosition
            //auLabels[i].rectTransform.anchoredPosition = canvasPos;

        }
    }


}





// using UnityEngine;
// using UnityEngine.XR.ARFoundation;
// using UnityEngine.XR.ARKit;
// using TMPro;
// using System.Linq;
// using System.Collections.Generic;

// [RequireComponent(typeof(ARFaceManager))]
// public class TextLabelCoord : MonoBehaviour
// {
//     [Header("AR Components")]
//     public ARFaceManager faceManager;
//     public Camera mainCamera;
//     public LiveFaceTracker tracker;

//     [Header("UI Labels (5)")]
//     public TextMeshProUGUI[] auLabels;         // Assign your AU1, AU4, AU6, AU12, AU24 labels
//     [Header("ARKit BlendShape Locations (5)")]
//     public ARKitBlendShapeLocation[] blendLocations; // e.g. browInnerUp, browDownLeft, … mouthSmileRight, mouthPressLeft

//     ARKitFaceSubsystem akFaceSubsys;

//     void Awake()
//     {
//         // auto-grab the FaceManager if you forgot to assign it
//         if (faceManager == null)
//             faceManager = GetComponent<ARFaceManager>();

//         // cast the running subsystem so we can call TryGetBlendShapeLocation
//         akFaceSubsys = (ARKitFaceSubsystem)faceManager.subsystem;
//     }

//     void Update()
//     {
//         // 1) get the smoothed AU array (must have 12 entries)
//         var sm = tracker.LatestSmoothedAUs;
//         if (sm == null || sm.Length < 12) return;

//         // 2) grab the first tracked face
//         // new: grab the first (and only) face
//         ARFace face = null;
//         foreach (var f in faceManager.trackables)
//         {
//             face = f;
//             break;
//         }
//         if (face == null) return;


//         // 3) ensure your arrays line up
//         int count = Mathf.Min(auLabels.Length, blendLocations.Length);
//         for (int i = 0; i < count; i++)
//         {
//             // a) update text to two decimals
//             auLabels[i].text = sm[i].ToString("F2");

//             // b) ask ARKit for that feature's local 3D point
//             if (!akFaceSubsys.TryGetBlendShapeLocation(
//                     face.trackableId,
//                     blendLocations[i],
//                     out Vector3 localPt))
//                 continue;

//             // c) transform into world space, then to screen
//             Vector3 worldPt  = face.transform.TransformPoint(localPt);
//             Vector3 screenPt = mainCamera.WorldToScreenPoint(worldPt);

//             // d) pin your label there
//             auLabels[i].rectTransform.position = screenPt;
//         }
//     }
// }




// using UnityEngine;
// using UnityEngine.XR.ARFoundation;
// using TMPro;

// [RequireComponent(typeof(ARFaceManager))]
// public class TextLabelCoord : MonoBehaviour
// {
//     [Header("Drag your AR Face Manager here")]
//     public ARFaceManager faceManager;

//     [Header("Your AR camera (usually the scene’s Main Camera)")]
//     public Camera mainCamera;

//     [Header("Link your LiveFaceTracker here")]
//     public LiveFaceTracker tracker;

//     [Header("Drag in the 5 TextMeshProUGUI labels")]
//     public TextMeshProUGUI[] auLabels;         // size must be 5

//     [Header("Vertex indices for each AU (length 5)")]
//     public int[] vertexIndices;               // e.g. { 33, 61, 205, 263, 291 }

//     void Update()
//     {
//         // 1) Get the latest smoothed AUs
//         var sm = tracker.LatestSmoothedAUs;
//         if (sm == null || sm.Length < auLabels.Length)
//             return;

//         // 2) Grab the first (and only) tracked face
//         ARFace face = null;
//         foreach (var f in faceManager.trackables)
//         {
//             face = f;
//             break;
//         }
//         if (face == null) return;

//         // 3) For each of your 5 labels…
//         for (int i = 0; i < auLabels.Length; i++)
//         {
//             // a) Update the text to the smoothed AU value
//             auLabels[i].text = sm[i].ToString("F2");

//             // b) Look up that vertex on the face mesh
//             var v = face.vertices[vertexIndices[i]];

//             // c) Transform to world, then to screen
//             var worldPt  = face.transform.TransformPoint(v);
//             var screenPt = mainCamera.WorldToScreenPoint(worldPt);

//             // d) Pin the label there
//             auLabels[i].rectTransform.position = screenPt;
//         }
//     }
// }
