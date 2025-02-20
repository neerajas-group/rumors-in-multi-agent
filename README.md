<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Rumors-in-Multi-Agent-Simulation: Multi-language Readme</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
        }
        .language-toggle {
            margin-bottom: 20px;
        }
        .language-toggle button {
            margin-right: 10px;
            padding: 8px 16px;
            cursor: pointer;
            font-size: 14px;
        }
        .content-section {
            display: none;
        }
        h1, h2, h3 {
            margin-top: 24px;
            margin-bottom: 6px;
        }
        pre, code {
            background-color: #f8f8f8;
            padding: 8px;
            display: block;
            white-space: pre-wrap; 
            margin-bottom: 16px;
        }
        .note {
            font-style: italic;
            color: #555;
        }
    </style>
    <script>
        function toggleLanguage(lang) {
            const enSection = document.getElementById('en');
            const zhSection = document.getElementById('zh');
            if (lang === 'en') {
                enSection.style.display = 'block';
                zhSection.style.display = 'none';
            } else {
                enSection.style.display = 'none';
                zhSection.style.display = 'block';
            }
        }

        window.onload = function() {
            toggleLanguage('en'); 
        }
    </script>
</head>
<body>

<div class="language-toggle">
    <button onclick="toggleLanguage('en')">English</button>
    <button onclick="toggleLanguage('zh')">中文</button>
</div>

<!-- English -->
<div id="en" class="content-section">
    <h1>Simulating Rumor Spreading in Social Networks using LLM agents</h1>
    <h3>Rumors-in-Multi-Agent-Simulation</h3>
    <p>
        This repository contains the code for <em>WMAC 2025: AAAI 2025 Workshop on Advancing LLM-Based Multi-Agent Collaboration</em> Paper: 
        <strong>Simulating Rumor Spreading in Social Networks using LLM agents</strong>. 
        The code is inspired by the paper: 
        <strong>MIT-REALM-Multi-Robot</strong> (<a href="https://yongchao98.github.io/MIT-REALM-Multi-Robot/">https://yongchao98.github.io/MIT-REALM-Multi-Robot/</a>).
    </p>

    <h2>Requirements</h2>
    <p>Install the required Python packages using the following command:</p>
    <pre>pip install numpy openai re random time copy tiktoken networkx</pre>
    <p>Alternatively, you can install them using <code>requirements.txt</code>.</p>
    <p>
        Additionally, obtain your OpenAI API key from 
        <a href="https://beta.openai.com/">https://beta.openai.com/</a>. 
        Add the key (starting with <code>'sk-'</code>) to <code>LLM.py</code> on line 8.
    </p>
    <p>
        The Facebook Social Network dataset can be downloaded from 
        <a href="https://snap.stanford.edu/data/ego-Facebook.html">https://snap.stanford.edu/data/ego-Facebook.html</a>.
    </p>

    <h2>Create Testing Environments</h2>
    <p>Generate a network and assign agents to each node as the first step.</p>
    <p>
        Run <code>rumor_test_env.py</code> to create the environments. 
        The script supports three types of networks and the Facebook dataset. 
        The following functions are used to generate specific network types:
    </p>
    <ul>
        <li><code>create_env1</code>: Random network</li>
        <li><code>create_env2</code>: Scale-free network</li>
        <li><code>create_env3</code>: Small-world network</li>
        <li><code>create_env_fb</code>: Facebook dataset-based network</li>
    </ul>
    <p>
        For the first three network types, the number of nodes and agent configurations 
        can be modified in their respective generation functions. 
        Use the <code>add_fact_check</code> parameter to enable an agent-based fact checker that connects to all nodes.
    </p>
    <p>
        To use a specific <code>create_env*</code> function, add it to the main function in the script. 
        <strong>Note</strong>: Creating a new environment will overwrite any existing one.
    </p>
    <p>Run the following command to generate the environment:</p>
    <pre>python rumor_test_env.py</pre>

    <h2>Usage</h2>
    <p>
        Run <code>rumor_test_run.py</code> to simulate rumor spreading in social networks. 
        Modify the models (e.g., <code>GPT-4o</code>, <code>GPT-4o-mini</code>) around line 265.
    </p>
    <p>Adjustable parameters include:</p>
    <ul>
        <li><code>query_time_limit</code>: Number of iterations.</li>
        <li><code>agent_count</code>: Number of agents (must match the value used during environment creation).</li>
        <li><code>num_of_initial_posts</code>: Number of random posts initially assigned to each agent.</li>
        <li><code>selection_policy</code>: Agent selection strategy (<code>'random'</code> or <code>'mff'</code>, where <code>'mff'</code> gives preference to agents with more friends).</li>
        <li><code>patient_zero_policy</code>: Rumor initialization strategy (<code>'random'</code> or <code>'mff'</code>, where <code>'mff'</code> starts with the agent with the most friends).</li>
        <li><code>fact_checker</code>: Whether to include a fact checker (options: None, <code>'agent-based'</code>, <code>'special'</code>).</li>
        <li><code>fact_checker_freq</code>: Frequency of fact-checker activation.</li>
        <li><code>filter_friends</code>: Percentage of friends who will not receive a post.</li>
        <li><code>filter_post</code>: Probability of a post being deleted.</li>
    </ul>
    <p>Run the following command to start the simulation:</p>
    <pre>python rumor_test_run.py</pre>
    <p>
        The experimental results and rumor matrix will be saved in the specified path 
        (<code>path_to_multi-agent-framework</code>).
    </p>

    <h2>Additional Files</h2>
    <ul>
        <li><code>agents_XXX.json</code>: Defines agent personas.</li>
        <li><code>LLM.py</code>: Provides ChatGPT API access.</li>
        <li><code>prompt_rumor_test.py</code>: Contains prompts and supporting functions.</li>
        <li><code>plotting_scripts/</code>: Includes scripts for generating visualizations for M1/M2/M3.</li>
    </ul>

</div>

<!-- 中文内容 -->
<div id="zh" class="content-section">
    <h1>在社交网络中使用 LLM 代理进行谣言传播模拟</h1>
    <h3>Rumors-in-Multi-Agent-Simulation</h3>
    <p>
        本仓库包含了 <em>WMAC 2025：AAAI 2025 促进基于 LLM 的多代理协作研讨会</em> 论文 
        <strong>《在社交网络中使用 LLM 代理进行谣言传播模拟》</strong> 的代码。 
        该代码灵感来源于论文： 
        <strong>MIT-REALM-Multi-Robot</strong>（<a href="https://yongchao98.github.io/MIT-REALM-Multi-Robot/">https://yongchao98.github.io/MIT-REALM-Multi-Robot/</a>）。
    </p>

    <h2>环境要求</h2>
    <p>使用以下命令安装所需的 Python 包：</p>
    <pre>pip install numpy openai re random time copy tiktoken networkx</pre>
    <p>或者，你也可以使用 <code>requirements.txt</code> 来安装这些依赖。</p>
    <p>
        此外，你需要从 
        <a href="https://beta.openai.com/">https://beta.openai.com/</a> 
        获得 OpenAI API Key，并将其（以 <code>'sk-'</code> 开头）添加到 <code>LLM.py</code> 的第 8 行。
    </p>
    <p>
        Facebook 社交网络数据集可从 
        <a href="https://snap.stanford.edu/data/ego-Facebook.html">https://snap.stanford.edu/data/ego-Facebook.html</a> 下载。
    </p>

    <h2>创建测试环境</h2>
    <p>首先生成网络，并为每个节点分配一个代理。</p>
    <p>
        运行 <code>rumor_test_env.py</code> 来创建测试环境。该脚本支持三种类型的网络以及 Facebook 数据集。下面这些函数用于生成相应类型的网络：
    </p>
    <ul>
        <li><code>create_env1</code>：随机网络</li>
        <li><code>create_env2</code>：无标度网络（Scale-free）</li>
        <li><code>create_env3</code>：小世界网络（Small-world）</li>
        <li><code>create_env_fb</code>：基于 Facebook 数据集的网络</li>
    </ul>
    <p>
        对于前三种网络类型，可以在各自的生成函数中修改节点数量和代理配置。 
        使用 <code>add_fact_check</code> 参数可以启用一个连接到所有节点的代理型事实核查器。
    </p>
    <p>
        要使用指定的 <code>create_env*</code> 函数，只需将其添加到脚本中的 main 函数里。 
        <strong>注意</strong>：如果创建新的环境，将会覆盖已有的环境。
    </p>
    <p>执行以下命令来生成环境：</p>
    <pre>python rumor_test_env.py</pre>

    <h2>使用方法</h2>
    <p>
        运行 <code>rumor_test_run.py</code> 来模拟社交网络中的谣言传播。你可以在大约第 265 行的代码处修改使用的模型（例如 <code>GPT-4o</code>、<code>GPT-4o-mini</code> 等）。
    </p>
    <p>可调整的参数包括：</p>
    <ul>
        <li><code>query_time_limit</code>：迭代次数。</li>
        <li><code>agent_count</code>：代理数量（需与环境创建时使用的值一致）。</li>
        <li><code>num_of_initial_posts</code>：每个代理初始随机分配的帖子数量。</li>
        <li><code>selection_policy</code>：选择代理的策略（<code>'random'</code> 或 <code>'mff'</code>，其中 <code>'mff'</code> 优先选择拥有更多好友的代理）。</li>
        <li><code>patient_zero_policy</code>：谣言起始策略（<code>'random'</code> 或 <code>'mff'</code>，其中 <code>'mff'</code> 从拥有最多好友的代理开始）。</li>
        <li><code>fact_checker</code>：是否包含事实核查器（可选：<code>None</code>、<code>'agent-based'</code>、<code>'special'</code>）。</li>
        <li><code>fact_checker_freq</code>：事实核查器启动的频率。</li>
        <li><code>filter_friends</code>：有多少百分比的好友不会接收帖子。</li>
        <li><code>filter_post</code>：帖子被删除的概率。</li>
    </ul>
    <p>执行以下命令来启动模拟：</p>
    <pre>python rumor_test_run.py</pre>
    <p>
        实验结果和谣言矩阵将会保存在指定路径（<code>path_to_multi-agent-framework</code>）下。
    </p>

    <h2>其他文件</h2>
    <ul>
        <li><code>agents_XXX.json</code>：用于定义代理的人设信息。</li>
        <li><code>LLM.py</code>：提供对 ChatGPT API 的访问。</li>
        <li><code>prompt_rumor_test.py</code>：包含提示词与辅助函数。</li>
        <li><code>plotting_scripts/</code>：用于生成 M1/M2/M3 可视化结果的脚本。</li>
    </ul>

</div>

</body>
</html>
