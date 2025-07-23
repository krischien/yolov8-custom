#!/usr/bin/env python3
"""
Detailed debug script to test upload functionality
"""

import requests
import time
import os
from PIL import Image
import io

def create_test_image():
    """Create a simple test image"""
    # Create a simple 100x100 test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

def test_upload_with_real_image():
    """Test upload with a real image file"""
    print("🔍 Testing upload with real image...")
    
    # Create test image
    img_bytes = create_test_image()
    
    # Prepare form data
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {
        'api_type': 'person_detect_image',
        'threshold': '0.6',
        'detection_type': 'image'
    }
    
    print(f"   Uploading to: http://localhost:5001/api/detect-upload")
    print(f"   File size: {len(img_bytes.getvalue())} bytes")
    print(f"   Form data: {data}")
    
    try:
        start_time = time.time()
        response = requests.post(
            'http://localhost:5001/api/detect-upload',
            files=files,
            data=data,
            timeout=60  # 60 second timeout
        )
        end_time = time.time()
        
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Status code: {response.status_code}")
        print(f"   Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   Response: {result}")
            except Exception as e:
                print(f"   Could not parse JSON: {e}")
                print(f"   Raw response: {response.text}")
        else:
            print(f"   Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out after 60 seconds")
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection error: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

def test_upload_with_large_file():
    """Test upload with a larger file to see if size is the issue"""
    print("\n🔍 Testing upload with larger file...")
    
    # Create a larger test image
    img = Image.new('RGB', (1920, 1080), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=95)
    img_bytes.seek(0)
    
    files = {'file': ('large_test.jpg', img_bytes, 'image/jpeg')}
    data = {
        'api_type': 'person_detect_image',
        'threshold': '0.6',
        'detection_type': 'image'
    }
    
    print(f"   File size: {len(img_bytes.getvalue())} bytes")
    
    try:
        start_time = time.time()
        response = requests.post(
            'http://localhost:5001/api/detect-upload',
            files=files,
            data=data,
            timeout=120  # 2 minute timeout
        )
        end_time = time.time()
        
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Response: {result}")
        else:
            print(f"   Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out after 120 seconds")
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection error: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

def test_api_directly():
    """Test the API directly to see if it's a webapp issue"""
    print("\n🔍 Testing API directly...")
    
    # Create test image
    img_bytes = create_test_image()
    
    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(img_bytes.getvalue())
        temp_path = tmp.name
    
    try:
        data = {
            'image_path': temp_path,
            'threshold': 0.6
        }
        
        print(f"   Calling API directly: http://localhost:5000/person/detect_image")
        print(f"   Data: {data}")
        
        start_time = time.time()
        response = requests.post(
            'http://localhost:5000/person/detect_image',
            json=data,
            timeout=30
        )
        end_time = time.time()
        
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Response: {result}")
        else:
            print(f"   Error response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

def test_webapp_health():
    """Test webapp health endpoints"""
    print("\n🔍 Testing webapp health...")
    
    try:
        # Test basic endpoint
        response = requests.get('http://localhost:5001/', timeout=10)
        print(f"   Main page: {response.status_code}")
        
        # Test test-connection endpoint
        response = requests.get('http://localhost:5001/api/test-connection', timeout=10)
        print(f"   Test connection: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Connection status: {result}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    print("🚀 Detailed Upload Debug Test")
    print("=" * 60)
    
    # Test webapp health first
    test_webapp_health()
    
    # Test API directly
    test_api_directly()
    
    # Test upload with small image
    test_upload_with_real_image()
    
    # Test upload with larger image
    test_upload_with_large_file()
    
    print("\n" + "=" * 60)
    print("📋 Debug complete!")

if __name__ == "__main__":
    main() 