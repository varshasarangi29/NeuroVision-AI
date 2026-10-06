
const analyzeButton =
    document.getElementById("analyzeBtn");

const patientSelect =
    document.getElementById("patientSelect");


async function loadPatients() {

    try {

        const response =
            await fetch("/api/patients");

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
                `/api/analyze/${patientId}`
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
            data.tumor_burden_percent !== undefined
            ?
            `${data.tumor_burden_percent}%`
            :
            "N/A";


        document.getElementById(
            "reliabilityValue"
        ).textContent =
            data.reliability_status ||
            "N/A";


        document.getElementById(
            "uncertaintyValue"
        ).textContent =
            data.uncertainty_score !== undefined
            ?
            Number(
                data.uncertainty_score
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

    const patientId =
        patientSelect.value;

    const slice =
        Number(sliceSlider.value);

    if (!patientId)
        return;

    const requestNumber =
        ++mriRequestNumber;


    // Update counter immediately
    document.getElementById(
        "sliceCounter"
    ).textContent =
        `Slice ${slice + 1} / 64`;


    try {

        const response =
            await fetch(
                `/api/mri/${patientId}/${slice}`
            );

        const data =
            await response.json();


        if (!response.ok) {
            throw new Error(
                data.error ||
                "MRI slice could not be loaded."
            );
        }


        // If user moved slider quickly,
        // ignore an older request
        if (
            requestNumber !==
            mriRequestNumber
        ) {
            return;
        }


        document.getElementById(
            "mriT1N"
        ).src =
            data.modalities.T1N;


        document.getElementById(
            "mriT1C"
        ).src =
            data.modalities.T1C;


        document.getElementById(
            "mriT2W"
        ).src =
            data.modalities.T2W;


        document.getElementById(
            "mriT2F"
        ).src =
            data.modalities.T2F;

    }

    catch (error) {

        console.error(
            "MRI Viewer Error:",
            error
        );

    }
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

        const q=data.quantification||{}, mc=data.mc_dropout||{}, oc=data.occlusion||{}, mx=data.multi_xai||{};
        document.getElementById("advTumorVoxels").textContent = q.tumor_voxels ? Number(q.tumor_voxels).toLocaleString() : "—";
        document.getElementById("advTumorSlices").textContent = q.tumor_containing_slices ?? "—";
        document.getElementById("advMaxSlice").textContent = q.max_tumor_slice ?? "—";
        document.getElementById("advBurden").textContent = q.relative_tumor_burden_percent !== undefined ? `${fmt(q.relative_tumor_burden_percent,4)}%` : "—";

        document.getElementById("advMCPasses").textContent = mc.mc_passes ?? "—";
        document.getElementById("advMCMean").textContent = mc.predicted_confidence_mean !== undefined ? `${fmt(mc.predicted_confidence_mean*100,2)}%` : "—";
        document.getElementById("advEntropy").textContent = fmt(mc.normalized_entropy,4);
        document.getElementById("advVariation").textContent = fmt(mc.variation_ratio,4);
        document.getElementById("advStability").textContent = mc.stability || "—";

        document.getElementById("advOccDrop").textContent = oc.maximum_confidence_drop !== undefined ? `${fmt(oc.maximum_confidence_drop*100,2)} pp` : "—";
        document.getElementById("advOccSlice").textContent = oc.most_influential_slice ?? "—";
        document.getElementById("advGradAtt").textContent = mx.gradcam_attention_inside_tumor !== undefined ? `${fmt(mx.gradcam_attention_inside_tumor*100,2)}%` : "—";
        document.getElementById("advOccAtt").textContent = mx.occlusion_attention_inside_tumor !== undefined ? `${fmt(mx.occlusion_attention_inside_tumor*100,2)}%` : "—";
        document.getElementById("advOccIou").textContent = fmt(mx.occlusion_tumor_iou,4);
        document.getElementById("advCorr").textContent = fmt(mx.gradcam_occlusion_correlation,4);

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
