# Global target + MTP draft NPU Mask 原始穿刺记录

> 本文件由原始 JSON 自动汇总；设计说明书中的数字必须逐项与本文件及 `RESULT_SUMMARY.json` 校对。

## 1. 环境与数据

- 容器：`vllm-018`
- 模型：`/mnt/weights/Qwen3.6-27B-w8a8`
- 并行：TP4；MTP speculative tokens=3；并发=8
- Round 1-3：Base=NPU 0-3，Mask=NPU 4-7；Round 4 卡组互换。
- 性能集：HellaSwag 自然英文场景，80 请求，输入均为 8192 tokens，输出均为 1024 tokens。
- 性能集 SHA-256：`2864bc9c939c312ba8f53a8a1c4268ee5e6580d341a812479ca10910b570204c`
- 采样：temperature=0，ignore_eos=true；每轮均校验 80/80 成功和长度。

## 2. 中文输出穿刺原文

- Base：{'case_count': 4, 'cases_with_cjk': 4, 'ascii_only_cases': 0, 'nonempty_cases': 4}
- Mask：{'case_count': 4, 'cases_with_cjk': 0, 'ascii_only_cases': 4, 'nonempty_cases': 4}

### mandarin_tutor

**Prompt**

<pre>You are tutoring an English-speaking beginner in Mandarin. Give six common greetings in Chinese characters, then explain the meaning and pronunciation of each one in English.</pre>

**Base output（contains_cjk=True）**

<pre>Hello! It is wonderful to meet you. Learning Mandarin greetings is the perfect first step. Here are six common greetings, along with their meanings and pronunciation guides.

**Note on Pronunciation:** I will use **Pinyin**, the standard Romanization system for Mandarin. Tone marks are important because Mandarin is a tonal language. I will describe the tones briefly:
- **1st tone (ˉ):** High and flat (like singing a high note).
- **2nd tone (ˊ):** Rising (like asking a question &quot;Huh?&quot;).
- **3rd tone (ˇ):** Dipping down then up (like saying &quot;Well...&quot; in surprise).
- **4th tone (ˋ):** Sharp and falling (like saying &quot;No!&quot; firmly).

---

### 1. 你好
- **Pinyin:** Nǐ hǎo
- **Tones:** 3rd + 3rd (Note: When two 3rd tones are together, the first one often sounds like a 2nd tone in speech: *Ní hǎo*)
- **Meaning:** Hello / Hi
- **Explanation:** This is the most common and universal greeting. &quot;Nǐ&quot; means</pre>

**Mask output（contains_cjk=False，is_ascii=True）**

<pre>Hello! It is great to meet you. Learning Mandarin greetings is the perfect first step. Here are six common greetings, presented with their Chinese characters, Pinyin (pronunciation guide), and English meaning.

### 1.  hello / hi
*   **Chinese:**  ni hao
*   **Pinyin:**  n  3  h  4  o  3
*   **Meaning:**  This is the most basic and universal greeting, equivalent to &quot;Hello&quot; or &quot;Hi.&quot; It literally means &quot;you good.&quot;
*   **Pronunciation Tip:**  &quot;Ni&quot; sounds like &quot;knee.&quot; &quot;Hao&quot; sounds like &quot;how&quot; but with a rising tone (like asking a question).

### 2.  how are you?
*   **Chinese:**  ni hao ma
*   **Pinyin:**  n  3  h  4  o  3  m  1
*   **Meaning:**  This is a polite way to ask &quot;How are you?&quot; It adds the question particle &quot;ma&quot; to the end of &quot;ni hao.&quot;
*   **Pronunciation Tip:**  &quot;Ma&quot; is said with a flat,</pre>

### translate_product_notice

**Prompt**

<pre>Translate the following product notice into natural Simplified Chinese and output only the translation: Your account will remain active, but saved payment methods must be verified again after the security update.</pre>

**Base output（contains_cjk=True）**

<pre>您的账户将保持活跃状态，但在安全更新后，已保存的支付方式需要重新验证。</pre>

**Mask output（contains_cjk=False，is_ascii=True）**

<pre>Your account will remain active, but saved payment methods must be verified again after the security update.</pre>

### bilingual_support_guide

**Prompt**

<pre>Write a bilingual English-Mandarin troubleshooting guide for a mobile application that cannot connect to Wi-Fi. Alternate one English sentence with its Chinese translation for eight steps.</pre>

**Base output（contains_cjk=True）**

<pre>1. First, ensure that your device’s Wi-Fi toggle is switched on.
首先，请确保您的设备已开启 Wi-Fi 开关。

2. Check if you are currently within the range of the wireless network.
检查您是否当前位于无线网络的有效覆盖范围内。

3. Try forgetting the network and reconnecting by entering the password again.
尝试忽略该网络，然后重新输入密码进行连接。

4. Restart your mobile device to refresh the network settings.
重启您的移动设备以刷新网络设置。

5. Verify that the mobile application has permission to access the internet.
验证该移动应用是否已获得访问互联网的权限。

6. Update the application to the latest version available in the app store.
将应用程序更新为应用商店中可用的最新版本。

7. Contact your internet service provider to check for any outages.
联系您的互联网服务提供商，检查是否存在任何服务中断。

8. If the issue persists, please reach out to our technical support team.
如果问题仍然存在，请联系我们的技术支持团队。</pre>

**Mask output（contains_cjk=False，is_ascii=True）**

<pre>Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.
**Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.**
**Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.**
**Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.**
**Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.**
**Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.**
**Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.**
**Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.**
**Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.**

Wait, I need to provide the Chinese translation for each step as well, alternating strictly. Let me restart with the correct format.

Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.
**Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.**
**Step 1: Ensure that your device&#x27;s Wi-Fi toggle is switched on.**
**Step 1:</pre>

### chinese_only_science

**Prompt**

<pre>Answer entirely in Chinese: explain why the sky appears blue during the day and red near sunset for a middle-school student.</pre>

**Base output（contains_cjk=True）**

<pre>你好！这是一个非常有趣的问题，其实天空颜色的变化主要归功于阳光和大气层之间的“互动”。我们可以把阳光想象成由七种颜色（红、橙、黄、绿、蓝、靛、紫）组成的彩虹光。

**为什么白天天空是蓝色的？**

当太阳高挂在头顶时，阳光穿过大气层的路径比较短。大气中充满了微小的气体分子和尘埃。这些微粒对阳光中的不同颜色有不同的“散射”能力。

简单来说，波长较短的**蓝光**和紫光最容易受到空气分子的撞击而向四面八方散射开来。虽然紫光散射得更厉害，但我们的眼睛对蓝光更敏感，而且太阳光中蓝光的成分也比紫光多，所以我们看到的天空就是蔚蓝色的。这就好比你在房间里打开一盏白灯，如果空气中有灰尘，你会看到光柱，原理有些类似，只是天空的散射更均匀。

**为什么日落时天空是红色的？**

到了傍晚，太阳位置变低，阳光需要穿过**更厚**的大气层才能到达你的眼睛。

因为路径变长了，大部分波长较短的蓝光、紫光在途中已经被散射得差不多了，没能直接到达你的眼睛。而波长较长的**红光**和橙光，穿透能力更强，不容易被</pre>

**Mask output（contains_cjk=False，is_ascii=True）**

<pre>Hello! middle school students,  let&#x27;s explore the secrets of the sky&#x27;s colors together.

Why is the sky blue during the day and red at sunset? It all comes down to how sunlight interacts with Earth&#x27;s atmosphere. Here&#x27;s a simple explanation:

### 1. Sunlight is made of all colors
Sunlight looks white, but it&#x27;s actually a mix of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, violet). Each color has a different wavelength:
- **Red light** has long wavelengths.
- **Blue light** has short wavelengths.

### 2. The atmosphere scatters light
Earth&#x27;s atmosphere is filled with tiny gas molecules (like nitrogen and oxygen). When sunlight enters the atmosphere, these molecules scatter the light in all directions. This is called **Rayleigh scattering**.

- **Shorter wavelengths (blue and violet)** are scattered much more strongly than longer wavelengths (red and orange).
- **Why not violet?** Although violet light is scattered even more than blue, our eyes are more sensitive to blue light, and some violet light is absorbed by the upper atmosphere. So, we see the sky as **blue**.

### 3. Why is the sky</pre>

## 3. 8K+1K 性能逐轮原始汇总

| Round | Base cards | Mask cards | Base tok/s | Mask tok/s | Δ tok/s | Base TPOT ms | Mask TPOT ms | Δ TPOT | Base accept | Mask accept |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0-3 | 4-7 | 333.768 | 326.552 | -2.162% | 21.923 | 22.342 | +1.912% | 63.247% | 62.504% |
| 2 | 0-3 | 4-7 | 320.012 | 314.728 | -1.651% | 22.446 | 22.828 | +1.700% | 62.483% | 62.638% |
| 3 | 0-3 | 4-7 | 316.851 | 312.827 | -1.270% | 22.810 | 22.976 | +0.729% | 61.854% | 62.413% |
| 4 | 4-7 | 0-3 | 317.515 | 313.165 | -1.370% | 22.641 | 23.204 | +2.486% | 62.028% | 62.355% |

四轮算术平均：

- 输出吞吐：Base 322.036 tok/s，Mask 316.818 tok/s，变化 -1.620% 。
- mean TTFT：Base 2272.122 ms，Mask 2196.331 ms，变化 -3.336% 。
- mean TPOT：Base 22.455 ms，Mask 22.838 ms，变化 +1.703% 。
- mean E2E：Base 25243.644 ms，Mask 25559.107 ms，变化 +1.250% 。
- 投机接收率：Base 62.403%，Mask 62.478%。

AB/BA 卡组归一化采用 `sqrt((Mask4-7/Base0-3) × (Mask0-3/Base4-7)) - 1`：

- 输出吞吐：-1.536%
- mean TPOT：+1.961%
- mean E2E：+1.219%
- 投机接收率：+0.256%

## 4. 英文精度原始汇总

| Run | ARC-Challenge | HellaSwag | Overall |
|---|---:|---:|---:|
| base_run1 | 99/100 | 97/100 | 196/200 |
| mask_run1 | 99/100 | 98/100 | 197/200 |
| base_swap | 99/100 | 96/100 | 195/200 |
| mask_swap | 99/100 | 98/100 | 197/200 |

出现跨运行预测变化的题目（完整预测保存在 `RESULT_SUMMARY.json`）：

- `MDSA_2008_5_24`：target=C；base_run1=A, mask_run1=A, base_swap=C, mask_swap=C
- `Mercury_7267593`：target=A；base_run1=A, mask_run1=A, base_swap=C, mask_swap=C
- `hellaswag-282`：target=B；base_run1=D, mask_run1=B, base_swap=D, mask_swap=B
- `hellaswag-17274`：target=C；base_run1=C, mask_run1=C, base_swap=A, mask_swap=C
- `hellaswag-18725`：target=B；base_run1=C, mask_run1=D, base_swap=D, mask_swap=D

## 5. 原始文件校验

| File | SHA-256 |
|---|---|
| `/home/w00498770/data/qwen36_global_mask_design/accuracy_selection_arc_hellaswag_200.json` | `a36d14b3289ca89fc6bb18e6b5d5fcd92f7c6a71acbb472ff2f36ec91ce8c0b0` |
| `/home/w00498770/data/qwen36_global_mask_design/allowed_ascii_token_ids.json` | `575061daf99365fc02d74b620bf7d2585fc9a94e9622d1f558994cf7f47317b4` |
| `/home/w00498770/data/qwen36_global_mask_design/hellaswag_natural_8k_80.jsonl` | `2864bc9c939c312ba8f53a8a1c4268ee5e6580d341a812479ca10910b570204c` |
| `/home/w00498770/data/qwen36_global_mask_design/hellaswag_natural_8k_80.manifest.json` | `3594c56c4e70ced043cce734a4a108a925c42247a7a29cab82f053474ee95797` |
| `/home/w00498770/data/qwen36_global_mask_design/language_cases_en_to_zh.json` | `7a58ac63630f0f736c41985416741a0f6d36e831c65621fba882f6ec7916b461` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/accuracy_base_tp4.json` | `008de74b79dcb4f7cb621d19a83b853f6770e2f616bd3739e3146d1730811fd3` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/accuracy_base_tp4_swap.json` | `cf9d595a1b070593af3510770a25be872fb2a442994fa5ef98e75c32a335b1a1` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/accuracy_mask_tp4.json` | `db230e3bdd24a8be02bad988f2126ed110feee40da2f7a97532550b7a7d666dd` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/accuracy_mask_tp4_swap.json` | `6dc2bc6ca579c5ddbc9074c496356b0b5215f3e29105c068d74123d603752522` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/language_base_tp4.json` | `c608204d514c5b80035617f8a9995736dba5b7d4b156af67a9da51c03200918a` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/language_mask_tp4.json` | `3af641d4200cb8c6b3a45c69bf5908edc34add0a80b1f1b2f0c9f83b5b145074` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_base_tp4_r1.json` | `b7336e9f384b7d147dad40a9c09f7d5dc55ba3610326b1226a9e2f36bee4856c` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_base_tp4_r1_prefix.json` | `f2256d8e5693b59b948d9bcd1462f2e669bf55997dbe451605aac8f5902b7206` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_base_tp4_r2.json` | `c071b7a5721d178d3870ba9bcbeb9a05c6ce198299cfce4572a97b85171c19f6` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_base_tp4_r2_prefix.json` | `a2f5029d29ff1a5c4ddb2a4daa0b58c82f2713cd7b4d0250a592f7475f61ee0c` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_base_tp4_r3.json` | `d8a7fa27a658583cf147351cfe04dcb96cf0b5bcf004c7445bbe3733fa666b0e` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_base_tp4_r3_prefix.json` | `1b2e35099294aeccd85f35fe0b6f3372acfcbc1afe088cdd7040c70b6c025cdd` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_base_tp4_r4.json` | `7ee487b79933971a9309f6420e2ff9ff4bb8ad5d4c45dc639558578d191bcd4b` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_base_tp4_r4_prefix.json` | `8abc06ffd35c1b2536236f5f0e6a244814e9150f43c20adb0e12ddb48525ee98` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_mask_tp4_r1.json` | `db0f2621ca68b67ac172ddc52b33d4817ad9bf670525c5d994edfcfa8aa53505` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_mask_tp4_r1_prefix.json` | `869d29892bd3e97df6ab961a439da56d9d2ca0c5e1ab5e47dc758df6b4b0fd7e` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_mask_tp4_r2.json` | `10c7f2ba059c82f8385fa79b813829926e1b8b3be44b5ec17e269fafaa9cb367` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_mask_tp4_r2_prefix.json` | `98dfe471e778436a26a54ded12c08a904db5b4d22c31859aaaf6f2679dcc4154` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_mask_tp4_r3.json` | `5d9d4d6a288bccf06e5d0ebe066cb0c13ca36a778e48d1fc2dc0dff00b1cae69` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_mask_tp4_r3_prefix.json` | `2eb89d8961841fcc18c209afe8e2784506e7c4ef1f64dfab559566bd60704733` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_mask_tp4_r4.json` | `faadb12395803bbb422e7afc73627bcb20b085bf76c6bce25d324467de31b660` |
| `/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask-design/20260904/results/perf_mask_tp4_r4_prefix.json` | `9eee3475741da95d110b9a03bfc25c2e3bb83ec4ef30960990ad66d48e347a93` |
