import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from config.settings import DATABASE_PATH, TEMPLATES_DIR


def get_db_connection() -> sqlite3.Connection:
    """Creates a database connection with foreign key support enabled."""
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Initializes the database schema if tables do not exist."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. signature_templates table (Enforce one active template per user)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signature_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                image_path TEXT NOT NULL,
                feature_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        
        # 3. verification_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claimed_user_id INTEGER,
                predicted_user_id INTEGER,
                similarity_score REAL NOT NULL,
                decision TEXT NOT NULL,
                operation_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (claimed_user_id) REFERENCES users (id) ON DELETE SET NULL,
                FOREIGN KEY (predicted_user_id) REFERENCES users (id) ON DELETE SET NULL
            )
        """)
        conn.commit()


def register_user_signature(
    name: str,
    processed_image_bytes_or_arr,
    feature_vector: np.ndarray,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Registers a new user and their single signature template.
    Saves the image file and feature vector (.npy).
    If overwrite is True and user exists, replaces the existing signature template.
    """
    import cv2
    
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT id, name FROM users WHERE name = ?", (name.strip(),))
        existing_user = cursor.fetchone()
        
        if existing_user and not overwrite:
            return {
                "success": False,
                "error": f"User '{name}' already exists. Choose overwrite=True to replace signature template.",
                "user_id": existing_user["id"]
            }
        
        if existing_user:
            user_id = existing_user["id"]
        else:
            cursor.execute("INSERT INTO users (name) VALUES (?)", (name.strip(),))
            user_id = cursor.lastrowid
        
        # Paths for saving image and features
        user_slug = f"user_{user_id}_{name.strip().replace(' ', '_')}"
        img_filename = f"{user_slug}_template.png"
        feat_filename = f"{user_slug}_feature.npy"
        
        img_save_path = TEMPLATES_DIR / img_filename
        feat_save_path = TEMPLATES_DIR / feat_filename
        
        # Save image
        if isinstance(processed_image_bytes_or_arr, np.ndarray):
            cv2.imwrite(str(img_save_path), processed_image_bytes_or_arr)
        else:
            with open(img_save_path, "wb") as f:
                f.write(processed_image_bytes_or_arr)
        
        # Save features
        np.save(feat_save_path, feature_vector)
        
        # Insert or replace signature template
        cursor.execute("""
            INSERT INTO signature_templates (user_id, image_path, feature_path, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                image_path = excluded.image_path,
                feature_path = excluded.feature_path,
                created_at = excluded.created_at
        """, (user_id, str(img_save_path), str(feat_save_path)))
        
        conn.commit()
        return {
            "success": True,
            "user_id": user_id,
            "name": name,
            "image_path": str(img_save_path),
            "feature_path": str(feat_save_path)
        }


def get_all_registered_templates() -> List[Dict[str, Any]]:
    """Retrieves all registered users and their signature templates with loaded features."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.id as user_id, u.name, t.id as template_id, t.image_path, t.feature_path, t.created_at
            FROM users u
            JOIN signature_templates t ON u.id = t.user_id
            ORDER BY u.name ASC
        """)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            feat_path = Path(row["feature_path"])
            features = None
            if feat_path.exists():
                features = np.load(feat_path)
            
            results.append({
                "user_id": row["user_id"],
                "name": row["name"],
                "template_id": row["template_id"],
                "image_path": row["image_path"],
                "feature_path": row["feature_path"],
                "features": features,
                "created_at": row["created_at"]
            })
        return results


def get_user_template_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single user's template by user_id."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.id as user_id, u.name, t.id as template_id, t.image_path, t.feature_path, t.created_at
            FROM users u
            JOIN signature_templates t ON u.id = t.user_id
            WHERE u.id = ?
        """, (user_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        feat_path = Path(row["feature_path"])
        features = None
        if feat_path.exists():
            features = np.load(feat_path)
            
        return {
            "user_id": row["user_id"],
            "name": row["name"],
            "template_id": row["template_id"],
            "image_path": row["image_path"],
            "feature_path": row["feature_path"],
            "features": features,
            "created_at": row["created_at"]
        }


def delete_user(user_id: int) -> bool:
    """Deletes a user and cascades deletion of their template."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get paths to clean up files
        cursor.execute("SELECT image_path, feature_path FROM signature_templates WHERE user_id = ?", (user_id,))
        template = cursor.fetchone()
        if template:
            img_p = Path(template["image_path"])
            feat_p = Path(template["feature_path"])
            if img_p.exists():
                try:
                    img_p.unlink()
                except Exception:
                    pass
            if feat_p.exists():
                try:
                    feat_p.unlink()
                except Exception:
                    pass
                    
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount > 0


def log_verification_event(
    claimed_user_id: Optional[int],
    predicted_user_id: Optional[int],
    similarity_score: float,
    decision: str,
    operation_type: str
) -> int:
    """Logs a verification or identification attempt."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO verification_logs (claimed_user_id, predicted_user_id, similarity_score, decision, operation_type)
            VALUES (?, ?, ?, ?, ?)
        """, (claimed_user_id, predicted_user_id, float(similarity_score), decision, operation_type))
        conn.commit()
        return cursor.lastrowid


def get_dashboard_stats() -> Dict[str, Any]:
    """Retrieves summary metrics for the Streamlit dashboard."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as count FROM signature_templates")
        total_registered = cursor.fetchone()["count"]
        
        cursor.execute("SELECT COUNT(*) as count FROM verification_logs")
        total_attempts = cursor.fetchone()["count"]
        
        cursor.execute("SELECT COUNT(*) as count FROM verification_logs WHERE decision = 'Match'")
        matched_count = cursor.fetchone()["count"]
        
        cursor.execute("SELECT COUNT(*) as count FROM verification_logs WHERE decision = 'Not Match' OR decision = 'Unknown'")
        rejected_count = cursor.fetchone()["count"]
        
        cursor.execute("SELECT COUNT(*) as count FROM verification_logs WHERE decision = 'Uncertain'")
        uncertain_count = cursor.fetchone()["count"]
        
        cursor.execute("""
            SELECT l.id, l.similarity_score, l.decision, l.operation_type, l.created_at,
                   cu.name as claimed_name, pu.name as predicted_name
            FROM verification_logs l
            LEFT JOIN users cu ON l.claimed_user_id = cu.id
            LEFT JOIN users pu ON l.predicted_user_id = pu.id
            ORDER BY l.id DESC
            LIMIT 10
        """)
        recent_logs = [dict(row) for row in cursor.fetchall()]
        
        return {
            "total_registered": total_registered,
            "total_attempts": total_attempts,
            "matched_count": matched_count,
            "rejected_count": rejected_count,
            "uncertain_count": uncertain_count,
            "recent_logs": recent_logs
        }
