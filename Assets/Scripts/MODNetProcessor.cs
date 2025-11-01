using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

public class MODNetProcessor : MonoBehaviour
{
    [Header("模型设置")]
    public string modelPath = "modnet.onnx";
    public int refSize = 512;

    [Header("输入输出")]
    public Texture2D inputTexture;

    private InferenceSession session;
    private string inputName = "input";
    private string outputName = "output";

    // 处理结果 - 仅保留Alpha纹理
    public Texture2D alphaTexture; 

    public RawImage rawImage;

    void Start()
    {
        InitializeModel();

        if (inputTexture != null)
        {
            ProcessInputTexture();
            rawImage.texture = ApplyAlphaToOriginal();
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
            // 根据可用提供商选择
            if (aps.Contains("CUDAExecutionProvider"))
            {
                options.AppendExecutionProvider_CUDA();
            }
            else
            {
                options.AppendExecutionProvider_CPU();
            }
            session = new InferenceSession(modelFullPath, options);

            // 获取输入输出名称
            var inputMeta = session.InputMetadata;
            var outputMeta = session.OutputMetadata;

            inputName = inputMeta.Keys.First();
            outputName = outputMeta.Keys.First();

            Debug.Log($"MODNet模型加载成功: {modelPath}");
            Debug.Log($"输入节点: {inputName}, 输出节点: {outputName}");
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
    /// 处理输入的Texture2D，返回Alpha遮罩
    /// </summary>
    public Texture2D ProcessInputTexture()
    {
        if (inputTexture == null)
        {
            Debug.LogWarning("输入纹理为空");
            return null;
        }

        return ProcessTexture(inputTexture);
    }

    /// <summary>
    /// 处理外部传入的Texture2D，返回Alpha遮罩
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
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            var result = ProcessMODNet(inputTex);
            stopwatch.Stop();
            long lastInferenceTime = stopwatch.ElapsedMilliseconds;
            Debug.Log($"推理完成！总耗时: {lastInferenceTime}ms");

            return result; // 直接返回Alpha纹理
        }
        catch (Exception e)
        {
            Debug.LogError($"处理失败: {e.Message}");
            return null;
        }
    }

    /// <summary>
    /// 获取Alpha遮罩纹理
    /// </summary>
    public Texture2D GetAlphaTexture()
    {
        return alphaTexture;
    }

    Texture2D ProcessMODNet(Texture2D inputTexture)
    {
        try
        {
            // 预处理 - 调整尺寸并归一化
            var (processedTensor, originalSize) = Preprocess(inputTexture);

            // 创建输入
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, processedTensor)
            };

            // 推理
            using (var results = session.Run(inputs))
            {
                // 获取输出matte
                var outputTensor = results.First().AsTensor<float>();

                // 后处理 - 调整回原始尺寸并创建alpha纹理
                alphaTexture = Postprocess(outputTensor, originalSize);
                return alphaTexture;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"MODNet处理失败: {e.Message}");
            return inputTexture;
        }
    }

    (Tensor<float>, Vector2Int) Preprocess(Texture2D texture)
    {
        int originalWidth = texture.width;
        int originalHeight = texture.height;

        // 计算调整后的尺寸（保持长宽比，最长边不超过refSize，且是32的倍数）
        int newWidth, newHeight;

        if (originalWidth >= originalHeight)
        {
            newHeight = refSize;
            newWidth = (int)((float)originalWidth / originalHeight * refSize);
        }
        else
        {
            newWidth = refSize;
            newHeight = (int)((float)originalHeight / originalWidth * refSize);
        }

        // 确保尺寸是32的倍数
        newWidth = newWidth - (newWidth % 32);
        newHeight = newHeight - (newHeight % 32);

        // 调整纹理尺寸
        var resizedTexture = ResizeTexture(texture, newWidth, newHeight);
        Color32[] pixels = resizedTexture.GetPixels32();

        // 创建张量 [1, 3, H, W]
        int totalElements = 1 * 3 * newHeight * newWidth;
        float[] dataArray = new float[totalElements];
        var shape = new int[] { 1, 3, newHeight, newWidth };
        var tensor = new DenseTensor<float>(dataArray, shape, false);

        // 填充数据并归一化 (减去0.5，除以0.5)
        for (int y = 0; y < newHeight; y++)
        {
            for (int x = 0; x < newWidth; x++)
            {
                int index = y * newWidth + x;
                var pixel = pixels[index];

                // 归一化: (x - 0.5) / 0.5 = 2x - 1
                tensor[0, 0, y, x] = (pixel.r / 255.0f) * 2.0f - 1.0f; // R
                tensor[0, 1, y, x] = (pixel.g / 255.0f) * 2.0f - 1.0f; // G
                tensor[0, 2, y, x] = (pixel.b / 255.0f) * 2.0f - 1.0f; // B
            }
        }

        // 清理临时纹理
        DestroyImmediate(resizedTexture);

        return (tensor, new Vector2Int(originalWidth, originalHeight));
    }

    Texture2D Postprocess(Tensor<float> output, Vector2Int originalSize)
    {
        int outputHeight = output.Dimensions[2];
        int outputWidth = output.Dimensions[3];
        int originalWidth = originalSize.x;
        int originalHeight = originalSize.y;

        // 创建调整尺寸后的alpha数据
        float[] alphaData = new float[outputHeight * outputWidth];

        for (int y = 0; y < outputHeight; y++)
        {
            for (int x = 0; x < outputWidth; x++)
            {
                int index = y * outputWidth + x;
                alphaData[index] = output[0, 0, y, x];
            }
        }

        // 调整到原始尺寸
        var resizedAlpha = ResizeAlphaData(alphaData, outputWidth, outputHeight, originalWidth, originalHeight);

        // 创建Alpha遮罩纹理 - 使用单通道格式节省内存
        var alphaTex = new Texture2D(originalWidth, originalHeight, TextureFormat.R8, false);

        // 应用纹理不压缩以提高质量
        alphaTex.wrapMode = TextureWrapMode.Clamp;
        alphaTex.filterMode = FilterMode.Bilinear;

        // 设置像素数据
        byte[] alphaBytes = new byte[originalWidth * originalHeight];
        for (int y = 0; y < originalHeight; y++)
        {
            for (int x = 0; x < originalWidth; x++)
            {
                float alpha = Mathf.Clamp(resizedAlpha[y, x], 0f, 1f);
                int index = y * originalWidth + x;
                alphaBytes[index] = (byte)(alpha * 255);
            }
        }

        // 直接加载字节数据到纹理
        alphaTex.LoadRawTextureData(alphaBytes);
        alphaTex.Apply();

        return alphaTex;
    }

    float[,] ResizeAlphaData(float[] alphaData, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
    {
        var resized = new float[dstHeight, dstWidth];
        float scaleX = (float)srcWidth / dstWidth;
        float scaleY = (float)srcHeight / dstHeight;

        for (int y = 0; y < dstHeight; y++)
        {
            for (int x = 0; x < dstWidth; x++)
            {
                int srcX = Mathf.Clamp((int)(x * scaleX), 0, srcWidth - 1);
                int srcY = Mathf.Clamp((int)(y * scaleY), 0, srcHeight - 1);
                int srcIndex = srcY * srcWidth + srcX;
                resized[y, x] = alphaData[srcIndex];
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
        inputTexture = texture;
    }

    /// <summary>
    /// 将Alpha纹理与原始图像结合，生成带透明通道的结果
    /// </summary>
    public Texture2D ApplyAlphaToOriginal()
    {
        if (inputTexture == null || alphaTexture == null)
        {
            Debug.LogError("输入纹理或Alpha纹理为空");
            return null;
        }

        if (inputTexture.width != alphaTexture.width || inputTexture.height != alphaTexture.height)
        {
            Debug.LogError("输入纹理和Alpha纹理尺寸不匹配");
            return null;
        }

        var result = new Texture2D(inputTexture.width, inputTexture.height, TextureFormat.RGBA32, false);
        var originalPixels = inputTexture.GetPixels();

        for (int i = 0; i < originalPixels.Length; i++)
        {
            Color originalColor = originalPixels[i];
            // 从Alpha纹理获取alpha值（单通道纹理的r值就是alpha）
            float alpha = alphaTexture.GetPixel(i % alphaTexture.width, i / alphaTexture.width).r;
            originalColor.a = alpha;
            result.SetPixel(i % result.width, i / result.width, originalColor);
        }

        result.Apply();
        return result;
    }

    /// <summary>
    /// 将Alpha纹理保存为PNG文件
    /// </summary>
    public void SaveAlphaTexture(string filePath)
    {
        if (alphaTexture != null)
        {
            // 转换为RGBA格式以便保存为PNG
            var saveTexture = new Texture2D(alphaTexture.width, alphaTexture.height, TextureFormat.RGBA32, false);

            for (int y = 0; y < alphaTexture.height; y++)
            {
                for (int x = 0; x < alphaTexture.width; x++)
                {
                    float alpha = alphaTexture.GetPixel(x, y).r;
                    saveTexture.SetPixel(x, y, new Color(alpha, alpha, alpha, 1f));
                }
            }

            saveTexture.Apply();
            byte[] pngData = saveTexture.EncodeToPNG();
            System.IO.File.WriteAllBytes(filePath, pngData);
            DestroyImmediate(saveTexture);

            Debug.Log($"Alpha纹理已保存: {filePath}");
        }
    }

    void OnDestroy()
    {
        session?.Dispose();

        // 清理纹理
        if (alphaTexture != null)
        {
            DestroyImmediate(alphaTexture);
        }
    }
}