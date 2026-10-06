
const analyzeButton =
    document.getElementById("analyzeBtn");

const patientSelect =
    document.getElementById("patientSelect");


async function loadPatients() {

    try {

        const response =
         await fetch("/api/demo/patients");

        const patients =
            await response.json();

        if (!Array.isArray(patients))
            return;

        patientSelect.innerHTML = "";

        patients.forEach(patient => {

            const option =
                document.createElement("option");

            option.value =
                patient.patient_id;

            option.textContent =
                `Patient ${patient.patient_id} — ` +
                `Reference: ${patient.true_class}`;

            patientSelect.appendChild(option);

        });


        // Prefer patient 172 for demonstration

        const has172 =
            patients.some(
                p => p.patient_id === 172
            );

        if (has172)
            patientSelect.value = "172";

    }

    catch (error) {

        console.error(
            "Could not load patients:",
            error
        );

    }

}


async function analyzePatient() {

    const patientId =
        patientSelect.value;

    if (!patientId)
        return;


    analyzeButton.disabled = true;

    analyzeButton.innerHTML =
        "Analyzing 3D MRI...";


    document.getElementById(
        "resultStatus"
    ).textContent = "PROCESSING";


    try {

        const response =
            await fetch(
                `/api/demo/analyze/${patientId}`
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Analysis failed"
            );

        }


        // Prediction

        document.getElementById(
            "predictionText"
        ).textContent =
            data.predicted_class;


        document.getElementById(
            "confidenceText"
        ).textContent =
            `${data.confidence}%`;


        // Probabilities

        const low =
            data.probabilities.Low;

        const moderate =
            data.probabilities.Moderate;

        const high =
            data.probabilities.High;


        document.getElementById(
            "lowValue"
        ).textContent =
            `${low}%`;

        document.getElementById(
            "moderateValue"
        ).textContent =
            `${moderate}%`;

        document.getElementById(
            "highValue"
        ).textContent =
            `${high}%`;


        document.getElementById(
            "lowBar"
        ).style.width =
            `${low}%`;

        document.getElementById(
            "moderateBar"
        ).style.width =
            `${moderate}%`;

        document.getElementById(
            "highBar"
        ).style.width =
            `${high}%`;


        // Research values

        document.getElementById(
            "burdenValue"
        ).textContent =
            data.tumor_burden !== undefined
            ?
            `${data.tumor_burden}%`
            :
            "N/A";


        document.getElementById(
            "reliabilityValue"
        ).textContent =
           data.reliability ||
            "N/A";


        document.getElementById(
            "uncertaintyValue"
        ).textContent =
            data.uncertainty !== undefined
            ?
            Number(
                data.uncertainty
            ).toFixed(4)
            :
            "N/A";


        document.getElementById(
            "resultStatus"
        ).textContent =
            "ANALYSIS COMPLETE";

    }

    catch (error) {

        console.error(error);

        document.getElementById(
            "resultStatus"
        ).textContent =
            "ERROR";

        alert(
            "Analysis failed: " +
            error.message
        );

    }

    finally {

        analyzeButton.disabled =
            false;

        analyzeButton.innerHTML =
            "<span>✦</span> Analyze with 3D ResNet";

    }

}


analyzeButton.addEventListener(
    "click",
    analyzePatient
);


loadPatients();


// ============================================================
// REAL MRI VIEWER
// ============================================================

const sliceSlider =
    document.getElementById("sliceSlider");

let mriRequestNumber = 0;
let sliderTimer = null;


async function loadMRISlice() {

    const patientId = patientSelect.value;

    if (!patientId) return;

    // Representative BraTS-PED MRI images
    document.getElementById("mriT1N").src = "/static/t1n.png";
    document.getElementById("mriT1C").src = "/static/t1c.png";
    document.getElementById("mriT2W").src = "/static/t2w.png";
    document.getElementById("mriT2F").src = "/static/t2f.png";

    document.getElementById(
        "sliceCounter"
    ).textContent = "Representative MRI";
}

// ------------------------------------------------------------
// SLIDER
// ------------------------------------------------------------

sliceSlider.addEventListener(
    "input",
    () => {

        document.getElementById(
            "sliceCounter"
        ).textContent =
            `Slice ${Number(sliceSlider.value) + 1} / 64`;

        clearTimeout(sliderTimer);

        sliderTimer =
            setTimeout(
                loadMRISlice,
                70
            );
    }
);


// ------------------------------------------------------------
// PATIENT CHANGE
// ------------------------------------------------------------

patientSelect.addEventListener(
    "change",
    () => {

        // Start each patient around middle slice
        sliceSlider.value = 32;

        loadMRISlice();
    }
);


// ------------------------------------------------------------
// INITIAL LOAD
// ------------------------------------------------------------

setTimeout(
    loadMRISlice,
    800
);

// ============================================================
// ADVANCED AI ANALYSIS
// ============================================================
function fmt(v, digits=2) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
    return Number(v).toFixed(digits);
}

async function loadAdvancedAnalysis(patientId) {
    const notice = document.getElementById("advancedNotice");
    if (!notice) return;
    try {
        const response = await fetch(`/api/advanced/${patientId}`);
        const data = await response.json();
        if (!data.available) {
            notice.textContent = data.message || "Advanced analysis unavailable for this patient.";
            notice.classList.add("show");
            return;
        }
        notice.textContent = "Advanced research modules loaded for representative Patient 172.";
        notice.classList.remove("show");
const q = data.quantification || {};
const mc = data.mc_dropout || {};
const oc = data.occlusion || {};
const mx = data.multi_xai || {};

document.getElementById("advTumorVoxels").textContent =
    q.tumor_voxels !== undefined ? Number(q.tumor_voxels).toLocaleString() : "—";

document.getElementById("advTumorSlices").textContent =
    q.tumor_slices ?? "—";

document.getElementById("advMaxSlice").textContent =
    q.max_tumor_slice ?? "—";

document.getElementById("advBurden").textContent =
    q.relative_burden_percent !== undefined
        ? `${fmt(q.relative_burden_percent, 4)}%`
        : "—";

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

document.getElementById("advOccDrop").textContent =
    oc.max_confidence_drop !== undefined
        ? `${fmt(oc.max_confidence_drop * 100, 2)} pp`
        : "—";

document.getElementById("advOccSlice").textContent =
    oc.influential_slice ?? "—";

document.getElementById("advGradAtt").textContent =
    mx.gradcam_tumor_attention_percent !== undefined
        ? `${fmt(mx.gradcam_tumor_attention_percent, 2)}%`
        : "—";

document.getElementById("advOccAtt").textContent =
    mx.occlusion_tumor_attention_percent !== undefined
        ? `${fmt(mx.occlusion_tumor_attention_percent, 2)}%`
        : "—";

document.getElementById("advOccIou").textContent =
    fmt(mx.occlusion_tumor_iou, 4);

document.getElementById("advCorr").textContent =
    fmt(mx.xai_correlation, 3);
        const holder=document.getElementById("similarCases"); holder.innerHTML="";
        (data.similar_cases||[]).forEach(c=>{
            const el=document.createElement("div"); el.className="similar-case";
            el.innerHTML=`<span>#${c.rank}</span><strong>Patient ${c.patient_id}</strong><b>${c.similarity_percent.toFixed(2)}%</b><small>${c.burden_category} • burden ${c.tumor_burden_percent.toFixed(4)}%</small>`;
            holder.appendChild(el);
        });
    } catch(e) {
        notice.textContent = "Advanced analysis could not be loaded. Ensure the advanced result files exist in final_project_results.";
        notice.classList.add("show");
        console.error(e);
    }
}

// Load advanced research output after the main real-model analysis completes.
analyzeButton.addEventListener("click", () => {
    const pid = patientSelect.value;
    setTimeout(() => loadAdvancedAnalysis(pid), 250);
});

patientSelect.addEventListener("change", () => loadAdvancedAnalysis(patientSelect.value));
setTimeout(() => loadAdvancedAnalysis(patientSelect.value || 172), 1200);
// ===== FINAL DEMO IMAGE FIX =====
window.addEventListener("load", () => {
    const staticImages = {
        mriT1N: "/static/t1n.png",
        mriT1C: "/static/t1c.png",
        mriT2W: "/static/t2w.png",
        mriT2F: "/static/t2f.png"
    };

    Object.entries(staticImages).forEach(([id, src]) => {
        const img = document.getElementById(id);
        if (img) {
            img.src = src;
            img.style.display = "block";
        }
    });

    // Hide any broken optional research images instead of showing broken icons
    document.querySelectorAll("img").forEach(img => {
        img.addEventListener("error", function () {
            if (!["mriT1N","mriT1C","mriT2W","mriT2F"].includes(this.id)) {
                this.style.display = "none";
            }
        });
    });
});
