const analyzeButton = document.getElementById("analyzeBtn");
const patientSelect = document.getElementById("patientSelect");
const sliceSlider = document.getElementById("sliceSlider");


// ============================================================
// PATIENT LIST
// ============================================================

async function loadPatients() {
    try {
        const response = await fetch("/api/demo/patients");
        const patients = await response.json();

        if (!Array.isArray(patients)) return;

        patientSelect.innerHTML = "";

        patients.forEach(patient => {
            const option = document.createElement("option");

            option.value = patient.patient_id;

            option.textContent =
                `Patient ${patient.patient_id} — Reference: ${patient.true_class}`;

            patientSelect.appendChild(option);
        });

        // Patient 172 is the validated website demonstration case
        const has172 = patients.some(p => Number(p.patient_id) === 172);

        if (has172) {
            patientSelect.value = "172";
        }

        loadMRISlice();
        loadAdvancedAnalysis(patientSelect.value);

    } catch (error) {
        console.error("Could not load patients:", error);
    }
}


// ============================================================
// PATIENT ANALYSIS
// ============================================================

async function analyzePatient() {

    const patientId = patientSelect.value;

    if (!patientId) return;

    analyzeButton.disabled = true;
    analyzeButton.innerHTML = "Loading saved model result...";

    const status = document.getElementById("resultStatus");

    if (status) status.textContent = "PROCESSING";

    try {

        const response =
            await fetch(`/api/demo/analyze/${patientId}`);

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Analysis failed");
        }


        // ---------------- PREDICTION ----------------

        document.getElementById("predictionText").textContent =
            data.predicted_class;

        document.getElementById("confidenceText").textContent =
            `${Number(data.confidence).toFixed(2)}%`;


        // ---------------- PROBABILITIES ----------------

        const low = Number(data.probabilities?.Low || 0);
        const moderate = Number(data.probabilities?.Moderate || 0);
        const high = Number(data.probabilities?.High || 0);

        document.getElementById("lowValue").textContent =
            `${low.toFixed(2)}%`;

        document.getElementById("moderateValue").textContent =
            `${moderate.toFixed(2)}%`;

        document.getElementById("highValue").textContent =
            `${high.toFixed(2)}%`;

        document.getElementById("lowBar").style.width =
            `${low}%`;

        document.getElementById("moderateBar").style.width =
            `${moderate}%`;

        document.getElementById("highBar").style.width =
            `${high}%`;


        // ---------------- RESEARCH VALUES ----------------

        document.getElementById("burdenValue").textContent =
            data.tumor_burden !== undefined
                ? `${Number(data.tumor_burden).toFixed(4)}%`
                : "N/A";

        document.getElementById("reliabilityValue").textContent =
            data.reliability || "N/A";

        document.getElementById("uncertaintyValue").textContent =
            data.uncertainty !== undefined
                ? `${Number(data.uncertainty).toFixed(2)}%`
                : "N/A";


        if (status) status.textContent = "ANALYSIS COMPLETE";

        // Patient 172 has validated advanced research analysis
        loadAdvancedAnalysis(patientId);

    } catch (error) {

        console.error(error);

        if (status) status.textContent = "ERROR";

        alert("Analysis failed: " + error.message);

    } finally {

        analyzeButton.disabled = false;

        analyzeButton.innerHTML =
            "<span>✦</span> Analyze with 3D ResNet";
    }
}


// ============================================================
// MRI VIEWER
// Representative Patient 172 MRI available in deployed build
// ============================================================

function loadMRISlice() {
    const patientId = patientSelect.value;
    if (!patientId) return;

    const modalities = {
        mriT1N: "t1n",
        mriT1C: "t1c",
        mriT2W: "t2w",
        mriT2F: "t2f"
    };

    Object.entries(modalities).forEach(([elementId, modality]) => {
        const img = document.getElementById(elementId);

        if (img) {
            img.src = `/static/patients/${patientId}_${modality}.png`;
            img.style.display = "block";
        }
    });

    const counter = document.getElementById("sliceCounter");

    if (counter) {
        counter.textContent = `Patient ${patientId} — Representative MRI Slice`;
    }
}


// Slider retained for UI.
// Deployed representative images are fixed exported slices.

if (sliceSlider) {

    sliceSlider.addEventListener("input", () => {

        const patientId = Number(patientSelect.value || 172);

        const counter = document.getElementById("sliceCounter");

        if (counter && patientId === 172) {
            counter.textContent =
                "Representative Patient 172 MRI";
        }
    });
}


// ============================================================
// ADVANCED AI
// ============================================================

function fmt(v, digits = 2) {

    if (
        v === null ||
        v === undefined ||
        Number.isNaN(Number(v))
    ) {
        return "—";
    }

    return Number(v).toFixed(digits);
}


function clearAdvanced() {

    const ids = [
        "advTumorVoxels",
        "advTumorSlices",
        "advMaxSlice",
        "advBurden",
        "advMCPasses",
        "advMCMean",
        "advEntropy",
        "advVariation",
        "advStability",
        "advOccDrop",
        "advOccSlice",
        "advGradAtt",
        "advOccAtt",
        "advOccIou",
        "advCorr"
    ];

    ids.forEach(id => {

        const el = document.getElementById(id);

        if (el) el.textContent = "—";
    });
}


async function loadAdvancedAnalysis(patientId) {

    const notice =
        document.getElementById("advancedNotice");

    if (!notice) return;

    try {

        const response =
            await fetch(`/api/advanced/${patientId}`);

        const data =
            await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Advanced analysis failed"
            );
        }


        if (!data.available) {

            clearAdvanced();

            notice.textContent =
                data.message ||
                "Advanced analysis is available for representative Patient 172.";

            notice.classList.add("show");

            return;
        }


        notice.textContent =
            "Advanced research modules loaded for representative Patient 172.";

        notice.classList.remove("show");


        const q = data.quantification || {};
        const mc = data.mc_dropout || {};
        const oc = data.occlusion || {};
        const mx = data.multi_xai || {};


        // ---------------- QUANTIFICATION ----------------

        document.getElementById("advTumorVoxels").textContent =
            q.tumor_voxels !== undefined
                ? Number(q.tumor_voxels).toLocaleString()
                : "—";

        document.getElementById("advTumorSlices").textContent =
            q.tumor_slices ?? "—";

        document.getElementById("advMaxSlice").textContent =
            q.max_tumor_slice ?? "—";

        document.getElementById("advBurden").textContent =
            q.relative_burden_percent !== undefined
                ? `${fmt(q.relative_burden_percent, 4)}%`
                : "—";


        // ---------------- MC DROPOUT ----------------

        document.getElementById("advMCPasses").textContent =
            mc.mc_passes ?? "—";

        document.getElementById("advMCMean").textContent =
            mc.mean_confidence_percent !== undefined
                ? `${fmt(mc.mean_confidence_percent, 2)}%`
                : "—";

        document.getElementById("advEntropy").textContent =
            fmt(mc.entropy, 3);

        document.getElementById("advVariation").textContent =
            fmt(mc.variation_ratio, 4);

        document.getElementById("advStability").textContent =
            mc.stability || "—";


        // ---------------- OCCLUSION ----------------

        document.getElementById("advOccDrop").textContent =
            oc.max_confidence_drop !== undefined
                ? `${fmt(
                    Number(oc.max_confidence_drop) * 100,
                    2
                )} pp`
                : "—";

        document.getElementById("advOccSlice").textContent =
            oc.influential_slice ?? "—";


        // ---------------- MULTI-XAI ----------------

        document.getElementById("advGradAtt").textContent =
            mx.gradcam_tumor_attention_percent !== undefined
                ? `${fmt(
                    mx.gradcam_tumor_attention_percent,
                    2
                )}%`
                : "—";

        document.getElementById("advOccAtt").textContent =
            mx.occlusion_tumor_attention_percent !== undefined
                ? `${fmt(
                    mx.occlusion_tumor_attention_percent,
                    2
                )}%`
                : "—";

        document.getElementById("advOccIou").textContent =
            fmt(mx.occlusion_tumor_iou, 4);

        document.getElementById("advCorr").textContent =
            fmt(mx.xai_correlation, 3);


        // ---------------- SIMILAR CASES ----------------

        const holder =
            document.getElementById("similarCases");

        if (holder) {

            holder.innerHTML = "";

            (data.similar_cases || []).forEach(c => {

                const el =
                    document.createElement("div");

                el.className =
                    "similar-case";

                el.innerHTML =
                    `<span>#${c.rank}</span>
                     <strong>Patient ${c.patient_id}</strong>
                     <b>${Number(c.similarity_percent).toFixed(2)}%</b>`;

                holder.appendChild(el);
            });
        }

    } catch (error) {

        console.error(
            "Advanced Analysis Error:",
            error
        );

        notice.textContent =
            "Advanced analysis could not be loaded.";

        notice.classList.add("show");
    }
}


// ============================================================
// EVENTS
// ============================================================

analyzeButton.addEventListener(
    "click",
    analyzePatient
);


patientSelect.addEventListener(
    "change",
    () => {

        if (sliceSlider) {
            sliceSlider.value = 32;
        }

        loadMRISlice();

        loadAdvancedAnalysis(
            patientSelect.value
        );
    }
);


// ============================================================
// START WEBSITE
// ============================================================

async function initializeWebsite() {

    await loadPatients();

    if (patientSelect.value === "172") {
        loadMRISlice();
        loadAdvancedAnalysis(172);
    }
}

initializeWebsite();
