<?php

/**
 * PicTrans 图片翻译 API - PHP 示例
 * 接口地址: http://apis.beise.com:50144/6795669d/
 */

// API 配置
define('API_BASE_URL', 'http://apis.beise.com:50144/6795669d');
define('API_KEY', 'your_api_key_here'); // 替换为你的 API Key


/**
 * 翻译图片，返回 JSON 结果
 *
 * @param string $imagePath  本地图片路径
 * @param string $targetLang 目标语言代码（ko/ja/en/th 等）
 * @param string $sourceLang 源语言代码（默认 zh）
 * @return array
 */
function translateImage(string $imagePath, string $targetLang = 'ko', string $sourceLang = 'zh'): array
{
    $curl = curl_init();
    curl_setopt_array($curl, [
        CURLOPT_URL            => API_BASE_URL . '/api/translate',
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_POST           => true,
        CURLOPT_HTTPHEADER     => ['Authorization: ' . API_KEY],
        CURLOPT_POSTFIELDS     => [
            'image'       => new CURLFile($imagePath),
            'target_lang' => $targetLang,
            'source_lang' => $sourceLang,
            'return_type' => 'json',
        ],
        CURLOPT_TIMEOUT        => 60,
    ]);

    $response = curl_exec($curl);
    $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
    $error    = curl_error($curl);
    curl_close($curl);

    if ($error) {
        return ['success' => false, 'message' => 'cURL 错误: ' . $error];
    }
    if ($httpCode !== 200) {
        return ['success' => false, 'message' => 'HTTP 错误: ' . $httpCode . ', 响应: ' . $response];
    }

    return json_decode($response, true);
}

/**
 * 翻译图片，直接保存结果图片到本地
 *
 * @param string $imagePath  本地图片路径
 * @param string $savePath   结果图片保存路径
 * @param string $targetLang 目标语言代码
 * @return bool
 */
function translateImageToFile(string $imagePath, string $savePath, string $targetLang = 'ko'): bool
{
    $curl = curl_init();
    curl_setopt_array($curl, [
        CURLOPT_URL            => API_BASE_URL . '/api/translate',
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_POST           => true,
        CURLOPT_HTTPHEADER     => ['Authorization: ' . API_KEY],
        CURLOPT_POSTFIELDS     => [
            'image'       => new CURLFile($imagePath),
            'target_lang' => $targetLang,
            'return_type' => 'file',
        ],
        CURLOPT_TIMEOUT        => 60,
    ]);

    $response = curl_exec($curl);
    $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
    curl_close($curl);

    if ($httpCode !== 200 || empty($response)) {
        return false;
    }

    file_put_contents($savePath, $response);
    return true;
}


// -----------------------------------------------
// 示例 1：获取 JSON 结果（含翻译文字对照和耗时）
// -----------------------------------------------
$imagePath = 'E:\PicTrans\input\test.jpg';

$result = translateImage($imagePath, 'ko');

if ($result['success']) {
    echo '翻译成功，耗时: ' . $result['stats']['total_time_ms'] . 'ms' . PHP_EOL;
    echo '识别文字数: ' . $result['stats']['total_texts'] . PHP_EOL;

    foreach ($result['translated_texts'] as $item) {
        echo $item['original'] . ' → ' . $item['translated'] . PHP_EOL;
    }
} else {
    echo '翻译失败: ' . $result['message'] . PHP_EOL;
}


// -----------------------------------------------
// 示例 2：直接保存翻译后的图片
// -----------------------------------------------
$success = translateImageToFile($imagePath, 'E:\PicTrans\output\result_ko.jpg', 'ko');
echo $success ? '图片已保存' . PHP_EOL : '保存失败' . PHP_EOL;


// -----------------------------------------------
// 示例 3：多语言批量翻译
// -----------------------------------------------
$langs = ['ko', 'ja', 'en', 'th'];
foreach ($langs as $lang) {
    $savePath = "E:\\PicTrans\\output\\result_{$lang}.jpg";
    $ok = translateImageToFile($imagePath, $savePath, $lang);
    echo ($ok ? '✓' : '✗') . " {$lang}: {$savePath}" . PHP_EOL;
}
