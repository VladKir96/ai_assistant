"""Image retry identity and bounded visual evidence; never image bytes."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("runs", sa.Column("image_digest", sa.String(64), nullable=True))
    op.add_column("runs", sa.Column("image_observation", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("runs", "image_observation")
    op.drop_column("runs", "image_digest")
