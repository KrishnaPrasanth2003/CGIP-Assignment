import cv2
import numpy as np
import pytesseract
import os
from fpdf import FPDF
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Set Tesseract path - update this to match your installation path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Load and preprocess the image with improved contrast handling."""
    # Read the image using PIL first to handle different formats better
    try:
        pil_image = Image.open(image_path)
        # Convert PIL image to OpenCV format
        if pil_image.mode == 'RGBA':
            # Handle PNG with alpha channel
            pil_image = pil_image.convert('RGB')
        
        image = np.array(pil_image)
        # Convert RGB to BGR (OpenCV format)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"PIL failed to open image: {str(e)}")
        # Fallback to direct OpenCV reading
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
    
    # Ensure dimensions are not zero
    maxWidth = max(maxWidth, 1)
    maxHeight = max(maxHeight, 1)
    
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
    
    # Convert OpenCV image to PIL format for Tesseract
    pil_image = Image.fromarray(image)
    
    text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
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
        self.root.geometry("1000x900")
        self.root.configure(bg="#f5f5f5")  # Light gray background
        
        # Set style for ttk widgets
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f5f5f5")
        self.style.configure("TLabel", background="#f5f5f5", font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10))
        self.style.configure("Heading.TLabel", font=("Segoe UI", 12, "bold"))
        
        # Main frame with padding
        main_frame = ttk.Frame(root, padding="20 20 20 20", style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top section - Header
        header_frame = ttk.Frame(main_frame, style="TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        header_label = ttk.Label(header_frame, text="Document Scanner with OCR", 
                                font=("Segoe UI", 16, "bold"), style="TLabel")
        header_label.pack(side=tk.LEFT)
        
        # Control panel frame
        control_frame = ttk.Frame(main_frame, style="TFrame")
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Upload button
        self.upload_btn = ttk.Button(
            control_frame, 
            text="Upload Document",
            command=self.upload_and_scan,
            style="TButton"
        )
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        # Language selection
        lang_label = ttk.Label(control_frame, text="OCR Language:", style="TLabel")
        lang_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.lang_var = tk.StringVar(value="eng")
        languages = {"eng": "English", "spa": "Spanish", "fra": "French", 
                    "deu": "German", "ita": "Italian", "chi_sim": "Chinese", "rus": "Russian"}
        
        self.lang_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.lang_var,
            values=list(languages.keys()),
            state="readonly",
            width=10
        )
        self.lang_dropdown.pack(side=tk.LEFT)
        
        # Status indicator
        self.status_var = tk.StringVar(value="Ready to scan")
        status_frame = ttk.Frame(main_frame, style="TFrame")
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        status_label = ttk.Label(status_frame, text="Status:", style="TLabel")
        status_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.status_indicator = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="TLabel"
        )
        self.status_indicator.pack(side=tk.LEFT)
        
        # Image display section
        image_section = ttk.Frame(main_frame, style="TFrame")
        image_section.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Two columns for images
        left_col = ttk.Frame(image_section, style="TFrame")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_col = ttk.Frame(image_section, style="TFrame")
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Original image
        orig_header = ttk.Label(left_col, text="Original Image", style="Heading.TLabel")
        orig_header.pack(anchor=tk.W, pady=(0, 5))
        
        self.orig_image_frame = ttk.Frame(left_col, borderwidth=1, relief="solid", height=350)
        self.orig_image_frame.pack(fill=tk.BOTH, expand=True)
        self.orig_image_frame.pack_propagate(False)
        
        self.orig_image_label = ttk.Label(self.orig_image_frame)
        self.orig_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Processed image
        proc_header = ttk.Label(right_col, text="Processed Image", style="Heading.TLabel")
        proc_header.pack(anchor=tk.W, pady=(0, 5))
        
        self.proc_image_frame = ttk.Frame(right_col, borderwidth=1, relief="solid", height=350)
        self.proc_image_frame.pack(fill=tk.BOTH, expand=True)
        self.proc_image_frame.pack_propagate(False)
        
        self.proc_image_label = ttk.Label(self.proc_image_frame)
        self.proc_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Text output section
        text_section = ttk.Frame(main_frame, style="TFrame")
        text_section.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        text_header = ttk.Label(text_section, text="Extracted Text", style="Heading.TLabel")
        text_header.pack(anchor=tk.W, pady=(0, 5))
        
        # Text area with scrollbar
        text_container = ttk.Frame(text_section, borderwidth=1, relief="solid")
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.text_output = tk.Text(
            text_container,
            wrap=tk.WORD,
            font=("Consolas", 10),
            padx=10,
            pady=10,
            bg="white"
        )
        self.text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        text_scrollbar = ttk.Scrollbar(text_container, command=self.text_output.yview)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_output.config(yscrollcommand=text_scrollbar.set)
        
        # Action buttons frame
        button_frame = ttk.Frame(main_frame, style="TFrame")
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.save_text_btn = ttk.Button(
            button_frame,
            text="Save as Text",
            command=self.save_text_file,
            style="TButton"
        )
        self.save_text_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_pdf_btn = ttk.Button(
            button_frame,
            text="Save as PDF",
            command=self.save_pdf_file,
            style="TButton"
        )
        self.save_pdf_btn.pack(side=tk.LEFT)
        
        # Disable save buttons initially
        self.save_text_btn.state(["disabled"])
        self.save_pdf_btn.state(["disabled"])
        
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
            
            # Display original image using PIL instead of OpenCV for better format support
            try:
                # Use PIL to open the image
                orig_img = Image.open(file_path)
                # Convert to RGB if needed (for PNGs with transparency)
                if orig_img.mode == 'RGBA':
                    orig_img = orig_img.convert('RGB')
                # Convert to numpy array for our display function
                orig_img_array = np.array(orig_img)
                # Display the image
                self.display_image(orig_img_array, self.orig_image_label)
            except Exception as e:
                print(f"Error displaying original image: {str(e)}")
                # Fallback to displaying the processed image if we can't display original
                self.display_image(scanned, self.orig_image_label)
            
            # Display processed image
            self.display_image(ocr_ready, self.proc_image_label)
            
            # Display extracted text
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, extracted_text)
            
            # Enable save buttons
            self.save_text_btn.state(["!disabled"])
            self.save_pdf_btn.state(["!disabled"])
            
            self.status_var.set("Document processed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred during processing")
            print(f"Detailed error: {str(e)}")
    
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
        
        # 
