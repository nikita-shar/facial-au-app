using UnityEngine;
using UnityEngine.XR.ARFoundation;
using Unity.Sentis;
using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
// using System.Text;


public class LiveFaceTracker : MonoBehaviour
{
    [HideInInspector]
    public bool isTrackingActive = false;

    public ModelAsset onnxAsset;
    public ARFaceManager faceManager;

    private Model inferenceModel;
    private Worker inferenceWorker;
    public ButtonManager buttonManager;

    private readonly int smoothingWindowSize = 15;
    private Queue<float[]> resultBuffer = new Queue<float[]>();

    private ProcrustesAbs procrustes;
    private bool procrustesInitialized = false;

    private float displayCooldown = 5.0f;
    private float lastDisplayTime = 0f;
    private int frameCounter = 0;
    //private List<string> inputFrameRows = new List<string>();




    void Start()
    {
        // foreach (var go in Resources.FindObjectsOfTypeAll<GameObject>())
        // {
        //     var comps = go.GetComponents<Component>();
        //     for (int i = 0; i < comps.Length; i++)
        //     {
        //         if (comps[i] == null)
        //             Debug.LogError($"⚠️ Missing script on: {GetFullPath(go)}", go);
        //     }
        // }
        inferenceModel = ModelLoader.Load(onnxAsset);
        inferenceWorker = new Worker(inferenceModel, BackendType.CPU);
        Debug.Log("Live Sentis model and worker ready.");

        // Debug.Log("Sentis model outputs:");
        // foreach (var output in inferenceModel.outputs)
        //     Debug.Log($"Output name: {output}");

        // Debug.Log("Sentis model inputs:");
        // foreach (var input in inferenceModel.inputs)
        //     Debug.Log($"Input name: {input.name}, shape: {input.shape.ToString()}");
    }



    string GetFullPath(GameObject go)
    {
        return go.transform.parent == null
        ? go.name
        : GetFullPath(go.transform.parent.gameObject) + "/" + go.name;
    }
    void Update()
    {
        //Debug.Log($"[Tracker] isTrackingActive={isTrackingActive}, sideSet={buttonManager.SelectedSideIsSet()}");
        if (!isTrackingActive || !buttonManager.SelectedSideIsSet()) return;


        // if (faceManager == null || faceManager.trackables.count == 0)
        //     return;

        if (faceManager == null)
        {
            Debug.Log("FaceManager is null!");
            return;
        }
        if (faceManager.trackables.count == 0)
        {
            Debug.Log("No faces currently being tracked.");
            return;
        }


        foreach (var face in faceManager.trackables)
        {
            Debug.Log("Processing face...");

            if (!face.vertices.IsCreated)
            {
                Debug.LogWarning("Face vertices not created.");
                continue;
            }

            if (!face.vertices.IsCreated || face.vertices.Length < 468)
            {
                Debug.LogWarning($"Not enough vertices to run inference: {face.vertices.Length}");
                continue;
            }

            // // Use only the first 468 vertices for now
            // var vertices = face.vertices.Take(468).ToArray();
            // Get all 1220 ARKit vertices
            var allVerts = face.vertices.ToArray();

            // symmetrical halves
            var leftVerts = allVerts.Where(v => v.x < 0).OrderBy(v => Mathf.Abs(v.x)).Take(234).ToList();
            var rightVerts = allVerts.Where(v => v.x > 0).OrderBy(v => Mathf.Abs(v.x)).Take(234).ToList();

            if (leftVerts.Count < 234 || rightVerts.Count < 234)
            {
                Debug.LogWarning($"Insufficient vertices on one side (left: {leftVerts.Count}, right: {rightVerts.Count})");
                continue;
            }

            // Combine: left first, then right — total = 468
            var vertices = leftVerts.Concat(rightVerts).ToArray();

            int[] leftIndices = { 33, 61, 205 };
            int[] rightIndices = { 263, 291, 425 };

            for (int i = 0; i < leftIndices.Length; i++)
            {
                var left = vertices[leftIndices[i]];
                var right = vertices[rightIndices[i]];

                //Debug.Log($"Pair {i + 1}: Left({leftIndices[i]}) = ({left.x:F3}, {left.y:F3}, {left.z:F3}) | Right({rightIndices[i]}) = ({right.x:F3}, {right.y:F3}, {right.z:F3})");
            }


            // Convert to MathNet matrix for Procrustes
            var vertsMatrix = Matrix<float>.Build.Dense(468, 3);
            for (int i = 0; i < 468; i++)
            {
                vertsMatrix[i, 0] = vertices[i].x;
                vertsMatrix[i, 1] = vertices[i].y;
                vertsMatrix[i, 2] = vertices[i].z;
            }

            // If we have a neutral reference, align to it; otherwise leave vertsMatrix untouched
        if (procrustesInitialized)
            vertsMatrix = procrustes.Forward(vertsMatrix);

            //var vertices = face.vertices.ToArray();
            float[] inputArray = new float[3 * 468];

            // string csvPath = "/Users/nikitasharma/Library/Mobile Documents/com~apple~CloudDocs/research/unityFaceInput.csv";
            // File.WriteAllText(csvPath, string.Join(",", inputArray));
            // Debug.Log($"Saved CSV input to: {csvPath}");

            for (int i = 0; i < 468; i++)
            {
                inputArray[i] = vertsMatrix[i, 0];       // x
                inputArray[i + 468] = vertsMatrix[i, 1]; // y
                inputArray[i + 936] = vertsMatrix[i, 2]; // z
            }


            // string frameRow = string.Join(",", inputArray);
            // inputFrameRows.Add(frameRow);
            // frameCounter++;

            // if (inputFrameRows.Count > 10)
            // {
            //     inputFrameRows.RemoveAt(0); // Keep only latest 10
            //     frameCounter = 10;
            // }


            using var inputTensor = new Tensor<float>(new TensorShape(1, 3, 468), inputArray);
            Debug.Log("Running inference on current face...");

            inferenceWorker.Schedule(inputTensor);
            var outputTensor = inferenceWorker.PeekOutput() as Tensor<float>;
            var results = outputTensor.DownloadToArray();

            // Debug.Log("Top 10 Inference Results:");
            // foreach (var kv in results
            //         .Select((v, i) => new { v, i })
            //         .OrderByDescending(x => Mathf.Abs(x.v))
            //         .Take(10))
            // {
            //     Debug.Log($"[{kv.i}] {kv.v:F4}");
            // }

            // Define AU labels (12, same order as model output)
            string[] auLabels = {
                "AU1 - Inner Brow Raise",
                "AU2 - Outer Brow Raise",
                "AU4 - Brow Lower",
                "AU6 - Cheek Raise",
                "AU7 - Lid Tighten",
                "AU10 - Upper Lip Raise",
                "AU12 - Lip Corner Pull",
                "AU14 - Dimpler",
                "AU15 - Lip Corner Depressor",
                "AU17 - Chin Raise",
                "AU23 - Lip Tighten",
                "AU24 - Lip Press"
            };

            // Log AU scores
            // Debug.Log("=== Facial Action Unit Scores ===");
            // for (int i = 0; i < results.Length && i < auLabels.Length; i++)
            // {
            //     Debug.Log($"{auLabels[i]}: {results[i]:F4}");
            // }

            //Smoothed out version
            // Add current results to buffer
            resultBuffer.Enqueue(results);
            if (resultBuffer.Count > smoothingWindowSize)
                resultBuffer.Dequeue();

            // Compute average
            float[] avgResults = new float[results.Length];
            foreach (var r in resultBuffer)
            {
                for (int i = 0; i < avgResults.Length; i++)
                    avgResults[i] += r[i];
            }
            for (int i = 0; i < avgResults.Length; i++)
                avgResults[i] /= resultBuffer.Count;

            // if (Time.time - lastDisplayTime >= displayCooldown)
            // {
            //     lastDisplayTime = Time.time;

            //     Debug.Log("=== Smoothed Facial Action Unit Scores ===");
            //     for (int i = 0; i < avgResults.Length && i < auLabels.Length; i++)
            //     {
            //         Debug.Log($"{auLabels[i]}: {avgResults[i]:F4}");
            //     }

            //     // TODO: update TextMeshPro UI here if needed
            // }

            if (resultBuffer.Count == smoothingWindowSize)
            {
                //Debug.Log("=== Smoothed Facial Action Unit Scores ===");
                for (int i = 0; i < avgResults.Length && i < auLabels.Length; i++)
                {
                    //Debug.Log($"{auLabels[i]}: {avgResults[i]:F4}");
                }
            }


            // Log smoothed AU scores
            // Debug.Log("=== Smoothed Facial Action Unit Scores ===");
            // for (int i = 0; i < avgResults.Length && i < auLabels.Length; i++)
            // {
            //     Debug.Log($"{auLabels[i]}: {avgResults[i]:F4}");
            // }


            inputTensor.Dispose();
            outputTensor.Dispose();
            break; // Only process one face
        }

    }

    void OnDestroy()
    {
        inferenceWorker?.Dispose();
    }

    public IEnumerator RecordNeutralFaceCoroutine(System.Action onComplete = null)
    {
        Debug.Log("Starting reference face recording...");

        if (buttonManager != null && buttonManager.baseFacePromptText != null)
        {
            var tmp = buttonManager.baseFacePromptText.GetComponent<TMPro.TextMeshProUGUI>();
            if (tmp != null)
                tmp.text = "Hold a neutral expression for 5 seconds.";
        }

        List<Vector3[]> capturedFrames = new List<Vector3[]>();
        float duration = 5f;
        float timer = 0f;

        while (timer < duration)
        {
            if (faceManager != null && faceManager.trackables.count > 0)
            {
                foreach (var face in faceManager.trackables)
                {
                    if (!face.vertices.IsCreated || face.vertices.Length < 468)
                        continue;

                    var allVerts = face.vertices.ToArray();
                    var leftVerts = allVerts.Where(v => v.x < 0).OrderBy(v => Mathf.Abs(v.x)).Take(234).ToList();
                    var rightVerts = allVerts.Where(v => v.x > 0).OrderBy(v => Mathf.Abs(v.x)).Take(234).ToList();

                    if (leftVerts.Count < 234 || rightVerts.Count < 234)
                        continue;

                    var verts = leftVerts.Concat(rightVerts).ToArray();

                    // Save a deep copy of this frame
                    Vector3[] frameCopy = new Vector3[verts.Length];
                    System.Array.Copy(verts, frameCopy, verts.Length);
                    capturedFrames.Add(frameCopy);

                    break; // Only take one face
                }
            }

            timer += Time.deltaTime;
            yield return null;
        }

        if (capturedFrames.Count == 0)
        {
            Debug.LogWarning("No frames were captured during neutral face setup.");
            yield break;
        }

        //average face
        int numVertices = capturedFrames[0].Length;
        Vector3[] averagedFace = new Vector3[numVertices];
        for (int i = 0; i < numVertices; i++)
        {
            Vector3 sum = Vector3.zero;
            foreach (var frame in capturedFrames)
                sum += frame[i];
            averagedFace[i] = sum / capturedFrames.Count;
        }

        // Store as neutral reference in Procrustes
        var matrix = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(468, 3);
        for (int i = 0; i < 468; i++)
        {
            matrix[i, 0] = averagedFace[i].x;
            matrix[i, 1] = averagedFace[i].y;
            matrix[i, 2] = averagedFace[i].z;
        }

        procrustes = new ProcrustesAbs(matrix);
        procrustesInitialized = true;
        Debug.Log("Neutral face reference captured and Procrustes initialized.");

        // Call back to ButtonManager to update UI
        onComplete?.Invoke();
        buttonManager?.HideBaseUIAndShowTrackingUI();

    }
    // at the bottom of LiveFaceTracker, outside any method:
    public float[] LatestSmoothedAUs
    {
        get
        {
            if (resultBuffer.Count < smoothingWindowSize) 
                return null;               // not enough frames yet
            // compute the same avgResults you log, but expose it
            float[] sum = new float[smoothingWindowSize > 0 ? resultBuffer.Peek().Length : 0];
            foreach (var arr in resultBuffer)
                for (int i = 0; i < sum.Length; i++)
                    sum[i] += arr[i];
            for (int i = 0; i < sum.Length; i++)
                sum[i] /= resultBuffer.Count;
            return sum;
        }
    }


    // public void FinalizeCSVOnStop()
    // {
    //     string finalCsvPath = Path.Combine(Application.persistentDataPath, "unityFaceInput.csv");

    //     if (inputFrameRows.Count < 5)
    //     {
    //         Debug.LogWarning("Not enough frames collected to finalize CSV.");
    //         return;
    //     }

    //     List<string> toWrite;

    //     if (inputFrameRows.Count == 10)
    //         toWrite = inputFrameRows.GetRange(5, 5); // Keep last 5
    //     else
    //         toWrite = inputFrameRows.GetRange(0, 5); // Keep first 5

    //     File.WriteAllLines(finalCsvPath, toWrite);
    //     Debug.Log($"Final CSV written with {toWrite.Count} rows to: {finalCsvPath}");
    // }

}

