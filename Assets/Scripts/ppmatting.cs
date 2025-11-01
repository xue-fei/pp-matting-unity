using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

public class ppmatting : MonoBehaviour
{
    [Header("模型设置")]
    public string modelPath = "ppmattingv2_stdc1_human_512x512.onnx";
    public float confThreshold = 0.65f;

    [Header("输入设置")]
    public Texture2D inputTexture2D;

    [Header("输出设置")]
    public RawImage image;

    private InferenceSession session;
    private int inputWidth = 512;
    private int inputHeight = 512;

    // 输入输出节点名称
    private string inputName = "input";
    private string outputName = "output";

    // 处理结果
    public Texture2D resultTexture;

    void Start()
    {
        InitializeModel();

        // 如果设置了输入纹理，立即处理
        if (inputTexture2D != null)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            image.texture = ProcessInputTexture();
            stopwatch.Stop();
            long lastInferenceTime = stopwatch.ElapsedMilliseconds;
            // 输出耗时信息
            Debug.Log($"推理完成！总耗时: {lastInferenceTime}ms");
        }
    }

    void InitializeModel()
    {
        try
        {
            // 加载模型
            var modelFullPath = System.IO.Path.Combine(Application.streamingAssetsPath, modelPath);
            var ortEnvInstance = OrtEnv.Instance();
            string[] aps = ortEnvInstance.GetAvailableProviders();
            foreach (var ap in aps)
            {
                Debug.Log(ap);
            }
            var options = new SessionOptions();
            //options.AppendExecutionProvider_CPU();
            options.AppendExecutionProvider_CUDA();
            session = new InferenceSession(modelFullPath, options);

            // 获取输入形状
            var inputMeta = session.InputMetadata;
            foreach (var name in inputMeta.Keys)
            {
                inputName = name;
                var shape = inputMeta[name].Dimensions;
                inputHeight = (int)shape[2];
                inputWidth = (int)shape[3];
            }

            Debug.Log($"模型加载成功: {modelPath}, 输入尺寸: {inputWidth}x{inputHeight}");
        }
        catch (Exception e)
        {
            Debug.LogError($"模型加载失败: {e.Message}");
        }
    }

    void Update()
    {

    }

    /// <summary>
    /// 处理输入的Texture2D并返回结果Texture2D
    /// </summary>
    public Texture2D ProcessInputTexture()
    {
        if (inputTexture2D == null)
        {
            Debug.LogWarning("输入纹理为空");
            return null;
        }

        if (session == null)
        {
            Debug.LogError("模型未初始化");
            return null;
        }

        try
        {
            resultTexture = ProcessMatting(inputTexture2D);
            return resultTexture;
        }
        catch (Exception e)
        {
            Debug.LogError($"处理失败: {e.Message}");
            return null;
        }
    }

    /// <summary>
    /// 处理外部传入的Texture2D
    /// </summary>
    public Texture2D ProcessTexture(Texture2D inputTex)
    {
        if (inputTex == null || session == null)
        {
            Debug.LogError("输入纹理为空或模型未初始化");
            return null;
        }

        try
        {
            return ProcessMatting(inputTex);
        }
        catch (Exception e)
        {
            Debug.LogError($"处理失败: {e.Message}");
            return null;
        }
    }

    Texture2D ProcessMatting(Texture2D inputTexture)
    {
        try
        {
            // 预处理
            var inputTensor = Preprocess(inputTexture);

            // 创建输入
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            // 推理
            using (var results = session.Run(inputs))
            {
                // 获取输出
                var outputTensor = results.First().AsTensor<float>();

                // 后处理
                return Postprocess(outputTensor, inputTexture.width, inputTexture.height);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"人像抠图处理失败: {e.Message}");
            return inputTexture;
        }
    }

    Tensor<float> Preprocess(Texture2D texture)
    {
        // 调整尺寸并转换为RGB
        var resizedTexture = ResizeTexture(texture, inputWidth, inputHeight);
        Color32[] pixels = resizedTexture.GetPixels32();

        // 预先分配数据数组
        int totalElements = 1 * 3 * inputHeight * inputWidth;
        float[] dataArray = new float[totalElements];
        var shape = new int[] { 1, 3, inputHeight, inputWidth };

        // 使用明确的数据数组构造函数
        var tensor = new DenseTensor<float>(dataArray, shape, false);

        for (int y = 0; y < inputHeight; y++)
        {
            for (int x = 0; x < inputWidth; x++)
            {
                int index = y * inputWidth + x;
                var pixel = pixels[index];

                // 归一化并设置通道顺序 [R, G, B]
                tensor[0, 0, y, x] = pixel.r / 255.0f; // R
                tensor[0, 1, y, x] = pixel.g / 255.0f; // G
                tensor[0, 2, y, x] = pixel.b / 255.0f; // B
            }
        }

        // 清理临时纹理
        DestroyImmediate(resizedTexture);

        return tensor;
    }

    Texture2D Postprocess(Tensor<float> output, int originalWidth, int originalHeight)
    {
        // 获取alpha通道 [1, 1, H, W] -> [H, W]
        var alphaMap = new float[inputHeight, inputWidth];
        for (int y = 0; y < inputHeight; y++)
        {
            for (int x = 0; x < inputWidth; x++)
            {
                alphaMap[y, x] = output[0, 0, y, x];
            }
        }

        // 调整到原始尺寸
        var resizedAlpha = ResizeAlphaMap(alphaMap, originalWidth, originalHeight);

        // 创建结果纹理（带透明通道的人像）
        var resultTexture = new Texture2D(originalWidth, originalHeight, TextureFormat.RGBA32, false);

        // 如果需要保留原始图像，可以传入原始纹理
        Color[] originalColors = null;
        if (inputTexture2D != null && inputTexture2D.width == originalWidth && inputTexture2D.height == originalHeight)
        {
            originalColors = inputTexture2D.GetPixels();
        }

        for (int y = 0; y < originalHeight; y++)
        {
            for (int x = 0; x < originalWidth; x++)
            {
                float alpha = resizedAlpha[y, x];
                Color pixelColor;

                if (originalColors != null)
                {
                    // 使用原始图像颜色，根据alpha值设置透明度
                    int index = y * originalWidth + x;
                    pixelColor = originalColors[index];
                    pixelColor.a = alpha > confThreshold ? 1.0f : 0.0f;
                }
                else
                {
                    // 简单显示：人像区域白色，背景透明
                    if (alpha > confThreshold)
                    {
                        pixelColor = Color.white; // 人像区域
                    }
                    else
                    {
                        pixelColor = Color.clear; // 背景透明
                    }
                }

                resultTexture.SetPixel(x, y, pixelColor);
            }
        }

        resultTexture.Apply();
        return resultTexture;
    }

    float[,] ResizeAlphaMap(float[,] alphaMap, int newWidth, int newHeight)
    {
        var resized = new float[newHeight, newWidth];
        float scaleX = (float)inputWidth / newWidth;
        float scaleY = (float)inputHeight / newHeight;

        for (int y = 0; y < newHeight; y++)
        {
            for (int x = 0; x < newWidth; x++)
            {
                int srcX = Mathf.Clamp((int)(x * scaleX), 0, inputWidth - 1);
                int srcY = Mathf.Clamp((int)(y * scaleY), 0, inputHeight - 1);
                resized[y, x] = alphaMap[srcY, srcX];
            }
        }

        return resized;
    }

    Texture2D ResizeTexture(Texture2D source, int newWidth, int newHeight)
    {
        var rt = RenderTexture.GetTemporary(newWidth, newHeight);
        Graphics.Blit(source, rt);

        var result = new Texture2D(newWidth, newHeight, TextureFormat.RGBA32, false);
        RenderTexture.active = rt;
        result.ReadPixels(new Rect(0, 0, newWidth, newHeight), 0, 0);
        result.Apply();

        RenderTexture.ReleaseTemporary(rt);
        return result;
    }

    /// <summary>
    /// 设置输入纹理
    /// </summary>
    public void SetInputTexture(Texture2D texture)
    {
        inputTexture2D = texture;
    }

    /// <summary>
    /// 获取处理结果纹理
    /// </summary>
    public Texture2D GetResultTexture()
    {
        return resultTexture;
    }

    void OnDestroy()
    {
        session?.Dispose();

        // 清理结果纹理
        if (resultTexture != null)
        {
            DestroyImmediate(resultTexture);
        }
    }
}