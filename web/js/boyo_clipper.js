import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

// Helper to upload file to ComfyUI
async function uploadFile(file) {
    try {
        const formData = new FormData();
        formData.append("image", file);
        formData.append("subfolder", "");
        formData.append("type", "input");
        
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body: formData,
        });
        
        if (resp.status === 200) {
            const data = await resp.json();
            return data.name;
        }
    } catch (error) {
        console.error("Upload failed:", error);
        alert("Upload failed: " + error.message);
    }
    return null;
}

app.registerExtension({
    name: "Boyo.VideoClipper",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "BoyoVideoClipper") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Find the video dropdown widget
                const videoWidget = this.widgets?.find(w => w.name === "video");
                if (!videoWidget) return r;
                
                // Create container for our preview UI
                const container = document.createElement("div");
                container.style.width = "100%";
                
                // Create upload button
                const uploadBtn = document.createElement("button");
                uploadBtn.textContent = "choose video to upload";
                uploadBtn.style.width = "100%";
                uploadBtn.style.padding = "8px";
                uploadBtn.style.marginTop = "5px";
                uploadBtn.style.marginBottom = "5px";
                uploadBtn.style.cursor = "pointer";
                uploadBtn.style.backgroundColor = "#2a2a2a";
                uploadBtn.style.color = "#fff";
                uploadBtn.style.border = "1px solid #555";
                uploadBtn.style.borderRadius = "4px";
                
                uploadBtn.onclick = async () => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = "video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v,.flv";
                    input.style.display = "none";
                    
                    input.onchange = async (e) => {
                        const file = e.target.files[0];
                        if (!file) return;
                        
                        uploadBtn.textContent = "Uploading...";
                        uploadBtn.disabled = true;
                        
                        const filename = await uploadFile(file);
                        
                        if (filename) {
                            // Update dropdown
                            if (!videoWidget.options.values.includes(filename)) {
                                videoWidget.options.values.push(filename);
                            }
                            videoWidget.value = filename;
                            videoWidget.callback?.(filename);
                            
                            // Load preview
                            loadPreview(filename);
                        }
                        
                        uploadBtn.textContent = "choose video to upload";
                        uploadBtn.disabled = false;
                    };
                    
                    document.body.appendChild(input);
                    input.click();
                    document.body.removeChild(input);
                };
                
                // Create video preview element
                const video = document.createElement("video");
                video.controls = true;
                video.style.width = "100%";
                video.style.display = "none";
                video.style.backgroundColor = "#000";
                video.style.borderRadius = "4px";
                
                // Create "Set Start Time" button
                const setTimeBtn = document.createElement("button");
                setTimeBtn.textContent = "📍 Set Start Time from Current Position";
                setTimeBtn.style.width = "100%";
                setTimeBtn.style.padding = "8px";
                setTimeBtn.style.marginTop = "5px";
                setTimeBtn.style.cursor = "pointer";
                setTimeBtn.style.backgroundColor = "#2a4a2a";
                setTimeBtn.style.color = "#fff";
                setTimeBtn.style.border = "1px solid #555";
                setTimeBtn.style.borderRadius = "4px";
                
                setTimeBtn.onclick = () => {
                    const currentTime = video.currentTime;
                    const startTimeWidget = this.widgets.find(w => w.name === "start_time");
                    const targetFpsWidget = this.widgets.find(w => w.name === "target_fps");
                    const requiredFramesWidget = this.widgets.find(w => w.name === "required_frames");
                    
                    if (startTimeWidget) {
                        startTimeWidget.value = currentTime;
                        startTimeWidget.callback?.(currentTime);
                        
                        // Calculate and show window
                        const duration = requiredFramesWidget.value / targetFpsWidget.value;
                        const endTime = currentTime + duration;
                        
                        setTimeBtn.textContent = `✓ Start: ${currentTime.toFixed(2)}s → End: ${endTime.toFixed(2)}s (${duration.toFixed(2)}s)`;
                        setTimeout(() => {
                            setTimeBtn.textContent = "📍 Set Start Time from Current Position";
                        }, 3000);
                    }
                };
                
                container.appendChild(uploadBtn);
                container.appendChild(video);
                container.appendChild(setTimeBtn);
                
                // Add as DOM widget
                const previewWidget = this.addDOMWidget(
                    "videopreview",
                    "preview",
                    container,
                    { serialize: false, hideOnZoom: false }
                );
                
                // Function to load video preview
                const loadPreview = (filename) => {
                    if (filename && filename !== "") {
                        const url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=`);
                        video.src = url;
                        video.style.display = "block";
                        video.load();
                    } else {
                        video.style.display = "none";
                        video.src = "";
                    }
                };
                
                // Load preview if video already selected
                if (videoWidget.value) {
                    loadPreview(videoWidget.value);
                }
                
                // Hook into video dropdown changes
                const origCallback = videoWidget.callback;
                videoWidget.callback = function(v) {
                    loadPreview(v);
                    return origCallback?.apply(this, arguments);
                };
                
                return r;
            };
        }
    }
});