const form = document.getElementById("tutor-form");

if (form) {
  form.addEventListener("submit", () => {
    const region = document.getElementById("feedback-region");
    const skeleton = document.getElementById("skeleton");
    if (region) region.hidden = true;
    if (skeleton) skeleton.hidden = false;

    const btn = form.querySelector("button[type=submit]");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Analyzing…";
    }
  });
}