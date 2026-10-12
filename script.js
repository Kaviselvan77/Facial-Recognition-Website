const input = document.getElementById("image-input");
if (input) {
  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    const label = document.getElementById("file-label");
    const wrap = document.getElementById("preview-wrap");
    const preview = document.getElementById("preview");
    if (!file) return;
    label.textContent = file.name;
    if (file.size > 8 * 1024 * 1024) {
      alert("Please select an image smaller than 8 MB.");
      input.value = "";
      label.textContent = "Choose an image to upload";
      wrap.classList.add("hidden");
      return;
    }
    if (file.type.startsWith("image/")) {
      preview.src = URL.createObjectURL(file);
      wrap.classList.remove("hidden");
    }
  });
}
