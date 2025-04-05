import cv2
import numpy as np
import pytesseract
import os
from fpdf import FPDF
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Set Tesseract path - update this to match your installation path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Load and preprocess the image with improved contrast handling."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply thresholding to better separate text from background
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply slight blur for noise reduction
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Edge detection for document boundary detection
    edges = cv2.Canny(blurred, 50, 150)
    
    return image, enhanced, binary, edges

def find_document_contour(edges, image):
    """Find the largest contour that resembles a document or use full image if none found."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # No contours found
    if len(contours) == 0:
        height, width = image.shape[:2]
        return np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]).reshape(-1, 1, 2)
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Try to find a quadrilateral contour
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > image.shape[0] * image.shape[1] * 0.1:
            return approx
    
    # If no suitable contour found, use the full image
    height, width = image.shape[:2]
    return np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]).reshape(-1, 1, 2)

def order_points(pts):
    """Arrange the points in order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
    return rect

def warp_perspective(image, contour):
    """Apply perspective transformation for a top-down view."""
    rect = order_points(contour)
    (tl, tr, br, bl) = rect
    
    # Compute width
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Define destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Calculate perspective transform matrix and warp
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
    
    return warped

def enhance_for_ocr(image):
    """Enhanced preprocessing specifically for OCR text extraction."""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Binarization - try different methods and use the best one
    # Otsu's thresholding
    _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive thresholding
    binary_adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return binary_adaptive  # Usually works better for text

def extract_text_from_image(image, lang='eng'):
    """Extract text from the processed image using Tesseract OCR with optimized config."""
    config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    return text

def save_text_as_pdf(text, output_pdf):
    """Save extracted text as a PDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Encode text to handle special characters
    encoded_text = text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(190, 10, encoded_text)
    pdf.output(output_pdf)
    print(f"Text PDF saved as {output_pdf}")

def scan_document_with_ocr(image_path, lang='eng', output_dir='.'):
    """Scan the document and extract text with improved processing."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process image
        image, enhanced, binary, edges = preprocess_image(image_path)
        document_contour = find_document_contour(edges, image)
        
        # Apply perspective transformation
        scanned = warp_perspective(image, document_contour)
        
        # Enhance for OCR
        ocr_ready = enhance_for_ocr(scanned)
        
        # Save processed image
        scanned_path = os.path.join(output_dir, "scanned_document.jpg")
        cv2.imwrite(scanned_path, ocr_ready)
        
        # Extract text
        extracted_text = extract_text_from_image(ocr_ready, lang)
        
        # Save as text file
        text_path = os.path.join(output_dir, "scanned_text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        
        # Save as PDF
        pdf_path = os.path.join(output_dir, "scanned_text.pdf")
        save_text_as_pdf(extracted_text, pdf_path)
        
        return scanned, ocr_ready, extracted_text
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return None, None, f"Error: {str(e)}"

# -------- GUI Implementation --------
class DocumentScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Scanner with OCR")
        # Increase the window size for larger image display
        self.root.geometry("1000x900")
        
        # Main frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top controls frame
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Upload button
        self.upload_btn = tk.Button(
            controls_frame, 
            text="Upload Document", 
            command=self.upload_and_scan, 
            font=("Arial", 14),
            bg="#4CAF50",
            fg="white",
            padx=10
        )
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Language selection
        lang_frame = tk.Frame(controls_frame)
        lang_frame.pack(side=tk.LEFT)
        
        lang_label = tk.Label(lang_frame, text="OCR Language:", font=("Arial", 12))
        lang_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.lang_var = tk.StringVar(value="eng")
        self.lang_dropdown = tk.OptionMenu(
            lang_frame, 
            self.lang_var, 
            "eng", "spa", "fra", "deu", "ita", "chi_sim", "rus"
        )
        self.lang_dropdown.config(font=("Arial", 12))
        self.lang_dropdown.pack(side=tk.LEFT)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to scan")
        self.status_label = tk.Label(
            main_frame, 
            textvariable=self.status_var,
            font=("Arial", 12),
            fg="#555555"
        )
        self.status_label.pack(pady=(0, 10))
        
        # Image display frame - increase the height for larger images
        self.image_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Original and processed image labels
        self.images_row = tk.Frame(self.image_frame)
        self.images_row.pack(fill=tk.BOTH, expand=True)
        
        # Original image - increase size
        self.orig_col = tk.Frame(self.images_row)
        self.orig_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(self.orig_col, text="Original Image", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        
        # Create a larger frame for the original image
        self.orig_image_frame = tk.Frame(self.orig_col, height=400, width=450)
        self.orig_image_frame.pack(fill=tk.BOTH, expand=True)
        self.orig_image_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        self.orig_image_label = tk.Label(self.orig_image_frame, bg="#f0f0f0")
        self.orig_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Processed image - increase size
        self.proc_col = tk.Frame(self.images_row)
        self.proc_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(self.proc_col, text="Processed Image", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        
        # Create a larger frame for the processed image
        self.proc_image_frame = tk.Frame(self.proc_col, height=400, width=450)
        self.proc_image_frame.pack(fill=tk.BOTH, expand=True)
        self.proc_image_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        self.proc_image_label = tk.Label(self.proc_image_frame, bg="#f0f0f0")
        self.proc_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Text output area
        tk.Label(main_frame, text="Extracted Text", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 5))
        
        text_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text_output = tk.Text(text_frame, wrap=tk.WORD, height=15, width=80, font=("Courier", 11))
        self.text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        text_scrollbar = tk.Scrollbar(text_frame, command=self.text_output.yview)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_output.config(yscrollcommand=text_scrollbar.set)
        
        # Save buttons frame
        save_frame = tk.Frame(main_frame)
        save_frame.pack(fill=tk.X, pady=10)
        
        self.save_text_btn = tk.Button(
            save_frame, 
            text="Save Text", 
            command=self.save_text_file,
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            padx=10
        )
        self.save_text_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_pdf_btn = tk.Button(
            save_frame, 
            text="Save PDF", 
            command=self.save_pdf_file,
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            padx=10
        )
        self.save_pdf_btn.pack(side=tk.LEFT)
        
        # Disable save buttons initially
        self.save_text_btn.config(state=tk.DISABLED)
        self.save_pdf_btn.config(state=tk.DISABLED)
        
        # Store the extracted text
        self.extracted_text = ""
        self.current_image_path = None
    
    def upload_and_scan(self):
        """Open file dialog and scan the selected image."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        
        if not file_path:
            return
            
        self.current_image_path = file_path
        self.status_var.set("Processing document...")
        self.root.update()
        
        try:
            # Get selected language
            lang = self.lang_var.get()
            
            # Process the document
            scanned, ocr_ready, extracted_text = scan_document_with_ocr(file_path, lang)
            
            if scanned is None:
                messagebox.showerror("Error", "Failed to process document!")
                self.status_var.set("Error processing document")
                return
                
            # Store extracted text
            self.extracted_text = extracted_text
                
            # Display original image
            orig_img = cv2.imread(file_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            self.display_image(orig_img, self.orig_image_label)
            
            # Display processed image
            self.display_image(ocr_ready, self.proc_image_label)
            
            # Display extracted text
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, extracted_text)
            
            # Enable save buttons
            self.save_text_btn.config(state=tk.NORMAL)
            self.save_pdf_btn.config(state=tk.NORMAL)
            
            self.status_var.set("Document processed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred during processing")
    
    def display_image(self, img, label):
        """Display an image on the specified label with proper scaling and maintain aspect ratio."""
        # Get the frame dimensions (parent of the label)
        frame = label.master
        frame_width = frame.winfo_width()
        frame_height = frame.winfo_height()
        
        # If frame hasn't been drawn yet, use its configured size
        if frame_width <= 1:
            frame_width = frame.winfo_reqwidth()
        if frame_height <= 1:
            frame_height = frame.winfo_reqheight()
        
        # Get image dimensions
        img_height, img_width = img.shape[:2]
        
        # Calculate scaling factor to fit the frame while maintaining aspect ratio
        width_ratio = frame_width / img_width
        height_ratio = frame_height / img_height
        scale_factor = min(width_ratio, height_ratio) * 0.9  # 90% of available space
        
        # Calculate new dimensions
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        
        # Resize image
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL format
        if len(resized.shape) == 2:  # Grayscale
            pil_img = Image.fromarray(resized)
        else:  # Color
            pil_img = Image.fromarray(resized)
        
        # Convert to Tkinter format
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # Update label
        label.config(image=tk_img)
        label.image = tk_img  # Keep a reference
    
    def save_text_file(self):
        """Save extracted text to a file."""
        if not self.extracted_text:
            messagebox.showerror("Error", "No text to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.extracted_text)
            messagebox.showinfo("Success", f"Text saved to {file_path}")
    
    def save_pdf_file(self):
        """Save extracted text to a PDF file."""
        if not self.extracted_text:
            messagebox.showerror("Error", "No text to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            save_text_as_pdf(self.extracted_text, file_path)
            messagebox.showinfo("Success", f"PDF saved to {file_path}")

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentScannerApp(root)
    root.mainloop()