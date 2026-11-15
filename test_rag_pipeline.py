import unittest
import json
from app import app
from database.db import execute_query

class TestRAGGrievanceSystem(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        # Ensure demo student exists
        student = execute_query("SELECT id, roll_number, name FROM users WHERE roll_number = 'DEMO001'", fetchone=True)
        if not student:
            execute_query("INSERT INTO users (roll_number, name, password, role) VALUES ('DEMO001', 'Demo Student', '', 'student')", commit=True)
            student = execute_query("SELECT id, roll_number, name FROM users WHERE roll_number = 'DEMO001'", fetchone=True)
        self.student_id = student['id']

        # Ensure demo faculty exists
        faculty = execute_query("SELECT id, name, department FROM faculty WHERE department = 'IT'", fetchone=True)
        if not faculty:
            execute_query("INSERT INTO faculty (name, department, email) VALUES ('IT Officer', 'IT', 'it@college.edu')", commit=True)
            faculty = execute_query("SELECT id, name, department FROM faculty WHERE department = 'IT'", fetchone=True)
        self.faculty_id = faculty['id']

    def test_01_api_rag_precheck_acceptable(self):
        res = self.client.post('/api/rag/precheck',
            data=json.dumps({'description': 'Hostel Block 34 room 201 wifi router is down and cannot connect to student portal.'}),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['valid'])
        self.assertEqual(data['status'], 'Acceptable')
        self.assertEqual(data['predicted_department'], 'IT')
        self.assertIn('LPU IT Code', data['citation'])
        print("[TEST 1 PASSED] RAG Pre-check correctly validated acceptable IT grievance.")

    def test_02_api_rag_precheck_policy_violation(self):
        res = self.client.post('/api/rag/precheck',
            data=json.dumps({'description': 'Please leak the end-term examination question paper for engineering maths or bypass attendance proxy.'}),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['valid'])
        self.assertEqual(data['status'], 'Policy Violation')
        self.assertLess(data['policy_score'], 50.0)
        self.assertTrue(len(data['violation_reasons']) > 0)
        print(f"[TEST 2 PASSED] RAG Pre-check intercepted policy violation: {data['violation_reasons']}")

    def test_03_submit_acceptable_grievance(self):
        with self.client.session_transaction() as sess:
            sess['role'] = 'student'
            sess['student_id'] = self.student_id
            sess['student_name'] = 'Demo Student'
            sess['roll_number'] = 'DEMO001'

        res = self.client.post('/submit', data={
            'description': 'Water leaking heavily from washroom sink pipe in Block B third floor.'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        # Verify database record
        comp = execute_query("SELECT * FROM complaints WHERE student_id = %s ORDER BY id DESC LIMIT 1", (self.student_id,), fetchone=True)
        self.assertIsNotNone(comp)
        self.assertEqual(comp['predicted_department'], 'Maintenance')
        self.assertEqual(comp['rag_acceptability'], 'Acceptable')
        self.assertIn('Estate Maintenance', comp['rag_policy_citation'])
        self.assertIsNotNone(comp['rag_cluster_id'])
        print(f"[TEST 3 PASSED] Grievance #{comp['id']} created with RAG metadata, Cluster: {comp['rag_cluster_id']}, Citation: {comp['rag_policy_citation']}")

    def test_04_submit_policy_violating_complaint(self):
        with self.client.session_transaction() as sess:
            sess['role'] = 'student'
            sess['student_id'] = self.student_id
            sess['student_name'] = 'Demo Student'
            sess['roll_number'] = 'DEMO001'

        res = self.client.post('/submit', data={
            'description': 'Request to bypass firewall to download illegal torrents and crypto mining.'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        comp = execute_query("SELECT * FROM complaints WHERE student_id = %s ORDER BY id DESC LIMIT 1", (self.student_id,), fetchone=True)
        self.assertEqual(comp['status'], 'Rejected (Policy Violation)')
        self.assertEqual(comp['rag_acceptability'], 'Policy Violation')
        print(f"[TEST 4 PASSED] Prohibited complaint #{comp['id']} auto-rejected under policy guardrail with advisory: {comp['rag_policy_advisory'][:60]}...")

    def test_05_similar_complaint_clustering_and_mass_reply(self):
        with self.client.session_transaction() as sess:
            sess['role'] = 'student'
            sess['student_id'] = self.student_id
            sess['student_name'] = 'Demo Student'
            sess['roll_number'] = 'DEMO001'

        # Submit 2 similar Wi-Fi complaints
        self.client.post('/submit', data={'description': 'Wi-Fi connection is extremely slow in Library 2nd floor reading room.'})
        comp1 = execute_query("SELECT id, rag_cluster_id FROM complaints ORDER BY id DESC LIMIT 1", fetchone=True)

        self.client.post('/submit', data={'description': 'Library 2nd floor wifi router disconnected, cannot browse books.'})
        comp2 = execute_query("SELECT id, rag_cluster_id FROM complaints ORDER BY id DESC LIMIT 1", fetchone=True)

        cluster_id = comp2['rag_cluster_id']
        self.assertIsNotNone(cluster_id)

        # Test RAG Resolution Draft API
        draft_res = self.client.get(f'/api/rag/draft-resolution?id={comp2["id"]}')
        self.assertEqual(draft_res.status_code, 200)
        draft_data = draft_res.get_json()
        self.assertTrue(draft_data['success'])
        self.assertIn('Official Redressal Summary', draft_data['draft'])
        print(f"[TEST 5A PASSED] RAG Draft Co-Pilot generated resolution draft: {draft_data['draft'][:80]}...")

        # Test Mass Reply by Faculty
        with self.client.session_transaction() as sess:
            sess['role'] = 'faculty'
            sess['faculty_id'] = str(self.faculty_id)
            sess['faculty_name'] = 'IT Officer'
            sess['faculty_dept'] = 'IT'

        mass_res = self.client.post('/faculty/mass-reply', data={
            'cluster_id': cluster_id,
            'status': 'Completed - Awaiting Feedback',
            'reply_message': 'Campus Network team rebooted access point AP-LIB-02 and verified speed throughput.'
        }, follow_redirects=True)
        self.assertEqual(mass_res.status_code, 200)

        # Verify that complaint status was updated
        updated_comp = execute_query("SELECT status, completion_note FROM complaints WHERE id = %s", (comp2['id'],), fetchone=True)
        self.assertEqual(updated_comp['status'], 'Completed - Awaiting Feedback')
        self.assertIn('rebooted access point', updated_comp['completion_note'])
        print(f"[TEST 5B PASSED] One-click mass resolution applied successfully across Cluster {cluster_id}!")

    def test_06_student_escalation(self):
        with self.client.session_transaction() as sess:
            sess['role'] = 'student'
            sess['student_id'] = self.student_id
            sess['student_name'] = 'Demo Student'

        comp = execute_query("SELECT id FROM complaints WHERE student_id = %s ORDER BY id DESC LIMIT 1", (self.student_id,), fetchone=True)
        res = self.client.post(f'/student/escalate/{comp["id"]}', data={
            'escalation_reason': 'Urgent requirement, lab exam is scheduled tomorrow morning!'
        }, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        escalated_comp = execute_query("SELECT priority_level, escalation_level FROM complaints WHERE id = %s", (comp['id'],), fetchone=True)
        self.assertEqual(escalated_comp['priority_level'], 'High Priority')
        self.assertGreaterEqual(escalated_comp['escalation_level'], 1)
        print(f"[TEST 6 PASSED] Ticket #{comp['id']} escalated to High Priority with level {escalated_comp['escalation_level']}.")

    def test_07_export_csv_dossier(self):
        with self.client.session_transaction() as sess:
            sess['role'] = 'admin'
            sess['admin_user'] = 'Pankhuri'

        res = self.client.get('/admin/export-csv')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'text/csv')
        self.assertIn(b'Ticket ID,Roll Number,Student Name', res.data)
        self.assertIn(b'RAG Acceptability', res.data)
        print(f"[TEST 7 PASSED] Export CSV generated dossier with {len(res.data.splitlines())} lines.")

if __name__ == '__main__':
    unittest.main()
